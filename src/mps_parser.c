/*
Copyright 2026 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "mps_parser.h"
#include "cuhsdlp.h"
#include "matrix_formats.h"
#include "utils.h"
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define READER_BUFFER_SIZE (4 * 1024 * 1024)
#define MPS_MAX_TOKENS 64
#define MPS_WARN_PRINT_LIMIT 5

typedef struct NameNode
{
    char *name;
    int index;
    struct NameNode *next;
} NameNode;

typedef struct
{
    NameNode **buckets;
    size_t num_buckets;
    size_t size;
} NameMap;

static unsigned long hash_string(const char *str)
{
    unsigned long hash = 5381;
    int c;
    while ((c = *str++))
    {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static void namemap_init(NameMap *map, size_t num_buckets)
{
    map->num_buckets = num_buckets;
    map->size = 0;
    map->buckets = safe_calloc(num_buckets, sizeof(NameNode *));
}

static void namemap_resize(NameMap *map)
{
    int old_num_buckets = map->num_buckets;
    NameNode **old_buckets = map->buckets;

    int new_num_buckets = old_num_buckets * 2;
    map->num_buckets = new_num_buckets;
    map->buckets = safe_calloc(new_num_buckets, sizeof(NameNode *));

    for (int i = 0; i < old_num_buckets; ++i)
    {
        NameNode *current = old_buckets[i];
        while (current)
        {
            NameNode *next = current->next;

            unsigned long h =
                hash_string(current->name) % (unsigned long)new_num_buckets;

            current->next = map->buckets[h];
            map->buckets[h] = current;

            current = next;
        }
    }

    free(old_buckets);
}

static void namemap_free(NameMap *map)
{
    if (!map || !map->buckets)
        return;
    for (size_t i = 0; i < map->num_buckets; ++i)
    {
        NameNode *current = map->buckets[i];
        while (current)
        {
            NameNode *to_free = current;
            current = current->next;
            free(to_free->name);
            free(to_free);
        }
    }
    free(map->buckets);
    memset(map, 0, sizeof(NameMap));
}

static int namemap_get(const NameMap *map, const char *name)
{
    unsigned long h = hash_string(name) % (unsigned long)map->num_buckets;
    for (NameNode *p = map->buckets[h]; p; p = p->next)
    {
        if (strcmp(p->name, name) == 0)
        {
            return p->index;
        }
    }
    return -1;
}

static int namemap_put(NameMap *map, const char *name)
{
    if (map->size >= map->num_buckets * 0.75)
    {
        namemap_resize(map);
    }

    unsigned long h = hash_string(name) % (unsigned long)map->num_buckets;

    for (NameNode *p = map->buckets[h]; p; p = p->next)
    {
        if (strcmp(p->name, name) == 0)
        {
            return p->index;
        }
    }

    NameNode *new_node = safe_malloc(sizeof(NameNode));

    new_node->name = strdup(name);
    if (!new_node->name)
    {
        free(new_node);
        return -1;
    }

    new_node->index = map->size++;
    new_node->next = map->buckets[h];
    map->buckets[h] = new_node;

    return new_node->index;
}

typedef struct
{
    bool is_gz;
    union FileHandle
    {
        gzFile gz_f;
        FILE *f;
    } handle;

    char *buffer;
    char *current_pos;
    char *end_pos;
    bool read_error;
} FastLineReader;

static FastLineReader *fast_reader_open(const char *filename)
{
    FastLineReader *reader = safe_calloc(1, sizeof(FastLineReader));

    reader->buffer = safe_malloc(READER_BUFFER_SIZE);

    if (strlen(filename) > 3 &&
        strcmp(filename + strlen(filename) - 3, ".gz") == 0)
    {
        reader->is_gz = true;
        reader->handle.gz_f = gzopen(filename, "rb");
        if (!reader->handle.gz_f)
        {
            free(reader->buffer);
            free(reader);
            return NULL;
        }
    }
    else
    {
        reader->is_gz = false;
        reader->handle.f = fopen(filename, "r");
        if (!reader->handle.f)
        {
            free(reader->buffer);
            free(reader);
            return NULL;
        }
    }

    reader->current_pos = reader->buffer;
    reader->end_pos = reader->buffer;

    return reader;
}

static void fast_reader_close(FastLineReader *reader)
{
    if (!reader)
        return;
    if (reader->is_gz)
    {
        if (reader->handle.gz_f)
            gzclose(reader->handle.gz_f);
    }
    else
    {
        if (reader->handle.f)
            fclose(reader->handle.f);
    }
    free(reader->buffer);
    free(reader);
}

static char *fast_reader_gets(FastLineReader *reader, char *line_buf,
                              int line_buf_size)
{
    int len = 0;

    while (1)
    {
        if (reader->current_pos >= reader->end_pos)
        {
            if (reader->is_gz)
            {
                int bytes_read =
                    gzread(reader->handle.gz_f, reader->buffer, READER_BUFFER_SIZE);
                if (bytes_read <= 0)
                {
                    int errnum = Z_OK;
                    gzerror(reader->handle.gz_f, &errnum);
                    if (bytes_read < 0 ||
                        (errnum != Z_OK && errnum != Z_STREAM_END))
                    {
                        fprintf(stderr,
                                "ERROR: Failed to read compressed input "
                                "(corrupt or truncated file).\n");
                        reader->read_error = true;
                        return NULL;
                    }
                    return (len > 0) ? line_buf : NULL;
                }
                reader->end_pos = reader->buffer + bytes_read;
            }
            else
            {
                size_t bytes_read =
                    fread(reader->buffer, 1, READER_BUFFER_SIZE, reader->handle.f);
                if (bytes_read == 0)
                {
                    if (ferror(reader->handle.f))
                    {
                        fprintf(stderr, "ERROR: Failed to read input file.\n");
                        reader->read_error = true;
                        return NULL;
                    }
                    return (len > 0) ? line_buf : NULL;
                }
                reader->end_pos = reader->buffer + bytes_read;
            }
            reader->current_pos = reader->buffer;
        }

        char *newline_pos = (char *)memchr(reader->current_pos, '\n',
                                           reader->end_pos - reader->current_pos);

        int bytes_to_copy;
        bool line_complete = (newline_pos != NULL);

        if (line_complete)
        {
            bytes_to_copy = newline_pos - reader->current_pos + 1;
        }
        else
        {
            bytes_to_copy = reader->end_pos - reader->current_pos;
        }

        if (len + bytes_to_copy >= line_buf_size)
        {
            fprintf(stderr, "Error: Line too long to fit in buffer.\n");
            reader->read_error = true;
            return NULL;
        }

        memcpy(line_buf + len, reader->current_pos, bytes_to_copy);
        len += bytes_to_copy;
        reader->current_pos += bytes_to_copy;
        line_buf[len] = '\0';

        if (line_complete)
        {
            return line_buf;
        }
    }
}

typedef struct
{
    int *row_indices;
    int *col_indices;
    double *values;
    size_t nnz;
    size_t capacity;
} CooMatrix;

typedef struct
{
    char *name;
    char type;
} BufferedRow;

typedef struct
{
    gzFile gz_file;
    FILE *file;
    bool is_gzipped;

    NameMap row_map;
    NameMap col_map;
    NameMap free_row_map;

    CooMatrix coo_matrix;
    BufferedRow *buffered_rows;
    size_t num_buffered_rows;
    size_t buffered_rows_capacity;

    char *constraint_types;
    double *objective_coeffs;
    double *var_lower_bounds;
    double *var_upper_bounds;
    double *constraint_lower_bounds;
    double *constraint_upper_bounds;

    size_t col_capacity;
    size_t constraint_capacity;

    char *objective_row_name;
    char *current_col_name;
    double objective_constant;
    objective_sense_t model_sense;
    int error_flag;

    int warn_unknown_row;
    int warn_unknown_col;
    int warn_bad_number;
    int warn_integrality;

} MpsParserState;

static void parser_warning(int *count, const char *fmt, ...)
{
    ++*count;
    if (*count > MPS_WARN_PRINT_LIMIT)
        return;
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "WARNING: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
    if (*count == MPS_WARN_PRINT_LIMIT)
        fprintf(stderr, "WARNING: further warnings of this kind suppressed.\n");
}

static bool token_has_prefix_ci(const char *token, const char *prefix)
{
    for (; *prefix != '\0'; ++token, ++prefix)
    {
        if (toupper((unsigned char)*token) != *prefix)
            return false;
    }
    return true;
}

static double parse_mps_number(MpsParserState *state, const char *token)
{
    char buf[64];
    size_t len = strlen(token);
    if (len >= sizeof(buf))
    {
        parser_warning(&state->warn_bad_number,
                       "invalid numeric field '%s' treated as 0.\n", token);
        return 0.0;
    }
    memcpy(buf, token, len + 1);
    for (size_t i = 0; i < len; ++i)
    {
        if (buf[i] == 'D' || buf[i] == 'd')
        {
            buf[i] = 'E';
            break;
        }
    }
    char *endptr;
    double value = strtod(buf, &endptr);
    if (endptr == buf || *endptr != '\0')
    {
        parser_warning(&state->warn_bad_number,
                       "invalid numeric field '%s' treated as 0.\n", token);
        return 0.0;
    }
    return value;
}

static int add_coo_entry(CooMatrix *coo, int row, int col, double value)
{
    if (coo->nnz >= coo->capacity)
    {
        size_t new_capacity = (coo->capacity == 0) ? 1024 : coo->capacity * 2;
        coo->row_indices =
            (int *)safe_realloc(coo->row_indices, new_capacity * sizeof(int));
        coo->col_indices =
            (int *)safe_realloc(coo->col_indices, new_capacity * sizeof(int));
        coo->values =
            (double *)safe_realloc(coo->values, new_capacity * sizeof(double));
        coo->capacity = new_capacity;
    }
    coo->row_indices[coo->nnz] = row;
    coo->col_indices[coo->nnz] = col;
    coo->values[coo->nnz] = value;
    coo->nnz++;
    return 0;
}

static bool ensure_column_capacity(MpsParserState *state)
{
    if (state->col_map.size < state->col_capacity)
    {
        return true;
    }

    size_t new_cap = (state->col_capacity == 0) ? 256 : state->col_capacity * 2;

    if (new_cap < state->col_capacity)
    {
        return false;
    }

    state->objective_coeffs =
        (double *)safe_realloc(state->objective_coeffs, new_cap * sizeof(double));
    state->var_lower_bounds =
        (double *)safe_realloc(state->var_lower_bounds, new_cap * sizeof(double));
    state->var_upper_bounds =
        (double *)safe_realloc(state->var_upper_bounds, new_cap * sizeof(double));

    for (size_t i = state->col_capacity; i < new_cap; ++i)
    {
        state->objective_coeffs[i] = 0.0;
        state->var_lower_bounds[i] = 0.0;
        state->var_upper_bounds[i] = INFINITY;
    }

    state->col_capacity = new_cap;
    return true;
}

static void free_parser_state(MpsParserState *state);
static int finalize_rows(MpsParserState *state);
static int parse_rows_section(MpsParserState *state, char **tokens,
                              int n_tokens);
static int parse_columns_section(MpsParserState *state, char **tokens,
                                 int n_tokens);
static int parse_rhs_section(MpsParserState *state, char **tokens,
                             int n_tokens);
static int parse_ranges_section(MpsParserState *state, char **tokens,
                                int n_tokens);
static int parse_bounds_section(MpsParserState *state, char **tokens,
                                int n_tokens);

typedef enum
{
    SEC_NONE,
    SEC_ROWS,
    SEC_COLUMNS,
    SEC_RHS,
    SEC_RANGES,
    SEC_BOUNDS,
    SEC_OBJSENSE,
    SEC_ENDATA
} MpsSection;

lp_problem_t *read_mps_file(const char *filename)
{
    MpsParserState state = {0};
    MpsSection current_section = SEC_NONE;
    bool rows_finalized = false;

    FastLineReader *reader = fast_reader_open(filename);
    if (!reader)
    {
        fprintf(stderr, "ERROR: Could not open file %s\n", filename);
        return NULL;
    }

    namemap_init(&state.row_map, 1024);
    namemap_init(&state.col_map, 1024);
    namemap_init(&state.free_row_map, 16);

    char line[4096];
    while (fast_reader_gets(reader, line, sizeof(line)))
    {
        if (state.error_flag)
            break;

        if (line[0] == '*' || line[0] == '\n' || line[0] == '\r')
            continue;

        char *tokens[MPS_MAX_TOKENS] = {NULL};
        int n_tokens = 0;
        char *saveptr;
        char *token = strtok_r(line, " \t\n\r", &saveptr);
        while (token != NULL && n_tokens < MPS_MAX_TOKENS)
        {
            tokens[n_tokens++] = token;
            token = strtok_r(NULL, " \t\n\r", &saveptr);
        }
        if (token != NULL)
        {
            fprintf(stderr, "ERROR: Too many fields on one line (limit %d).\n",
                    MPS_MAX_TOKENS);
            state.error_flag = 1;
            break;
        }
        if (n_tokens == 0)
            continue;

        bool objsense_inline =
            n_tokens == 2 && strcmp(tokens[0], "OBJSENSE") == 0 &&
            (token_has_prefix_ci(tokens[1], "MIN") ||
             token_has_prefix_ci(tokens[1], "MAX"));
        if (isalpha((unsigned char)tokens[0][0]) &&
            (n_tokens == 1 || objsense_inline))
        {
            MpsSection next_section = SEC_NONE;
            if (strcmp(tokens[0], "ROWS") == 0)
                next_section = SEC_ROWS;
            else if (strcmp(tokens[0], "COLUMNS") == 0)
                next_section = SEC_COLUMNS;
            else if (strcmp(tokens[0], "RHS") == 0)
                next_section = SEC_RHS;
            else if (strcmp(tokens[0], "RANGES") == 0)
                next_section = SEC_RANGES;
            else if (strcmp(tokens[0], "BOUNDS") == 0)
                next_section = SEC_BOUNDS;
            else if (strcmp(tokens[0], "OBJSENSE") == 0)
                next_section = SEC_OBJSENSE;
            else if (strcmp(tokens[0], "ENDATA") == 0)
            {
                next_section = SEC_ENDATA;
            }

            if (next_section != SEC_NONE)
            {
                if (current_section == SEC_ROWS && next_section != SEC_ROWS &&
                    !rows_finalized)
                {
                    if (finalize_rows(&state) != 0)
                        state.error_flag = 1;
                    rows_finalized = true;
                }

                current_section = next_section;
                if (current_section == SEC_ENDATA)
                    break;
                if (current_section == SEC_OBJSENSE && n_tokens == 2 &&
                    token_has_prefix_ci(tokens[1], "MAX"))
                {
                    state.model_sense = OBJECTIVE_SENSE_MAXIMIZE;
                }
                continue;
            }
        }

        switch (current_section)
        {
        case SEC_OBJSENSE:
            if (token_has_prefix_ci(tokens[0], "MAX"))
            {
                state.model_sense = OBJECTIVE_SENSE_MAXIMIZE;
            }
            else if (!token_has_prefix_ci(tokens[0], "MIN"))
            {
                fprintf(stderr,
                        "WARNING: unrecognized OBJSENSE value '%s' ignored "
                        "(minimize assumed).\n", tokens[0]);
            }
            break;
        case SEC_ROWS:
            if (parse_rows_section(&state, tokens, n_tokens) != 0)
                state.error_flag = 1;
            break;
        case SEC_COLUMNS:
            if (parse_columns_section(&state, tokens, n_tokens) != 0)
                state.error_flag = 1;
            break;
        case SEC_RHS:
            if (parse_rhs_section(&state, tokens, n_tokens) != 0)
                state.error_flag = 1;
            break;
        case SEC_RANGES:
            if (parse_ranges_section(&state, tokens, n_tokens) != 0)
                state.error_flag = 1;
            break;
        case SEC_BOUNDS:
            if (parse_bounds_section(&state, tokens, n_tokens) != 0)
                state.error_flag = 1;
            break;
        default:
            break;
        }
    }

    if (current_section == SEC_ROWS && !rows_finalized && !state.error_flag)
    {
        if (finalize_rows(&state) != 0)
            state.error_flag = 1;
    }

    if (reader->read_error)
        state.error_flag = 1;
    fast_reader_close(reader);

    if (state.error_flag)
    {
        fprintf(stderr, "ERROR: Failed to parse MPS file.\n");
        free_parser_state(&state);
        return NULL;
    }

    lp_problem_t *prob = safe_calloc(1, sizeof(lp_problem_t));

    prob->num_variables = state.col_map.size;
    prob->num_constraints = state.row_map.size;
    prob->objective_constant = state.objective_constant;
    prob->objective_sense = state.model_sense;

    prob->objective_vector = state.objective_coeffs;
    prob->variable_lower_bound = state.var_lower_bounds;
    prob->variable_upper_bound = state.var_upper_bounds;
    prob->constraint_lower_bound = state.constraint_lower_bounds;
    prob->constraint_upper_bound = state.constraint_upper_bounds;

    prob->primal_start = NULL;
    prob->dual_start = NULL;

    state.objective_coeffs = NULL;
    state.var_lower_bounds = NULL;
    state.var_upper_bounds = NULL;
    state.constraint_lower_bounds = NULL;
    state.constraint_upper_bounds = NULL;

    matrix_desc_t desc = {0};
    desc.m = prob->num_constraints;
    desc.n = prob->num_variables;
    desc.fmt = matrix_coo;
    desc.data.coo.nnz = (int)state.coo_matrix.nnz;
    desc.data.coo.row_ind = state.coo_matrix.row_indices;
    desc.data.coo.col_ind = state.coo_matrix.col_indices;
    desc.data.coo.vals = state.coo_matrix.values;

    if (coo_to_csr(&desc,
                   &prob->constraint_matrix_row_pointers,
                   &prob->constraint_matrix_col_indices,
                   &prob->constraint_matrix_values,
                   &prob->constraint_matrix_num_nonzeros) != 0)
    {
        fprintf(stderr, "ERROR: Failed to convert matrix to CSR format.\n");
        lp_problem_free(prob);
        free_parser_state(&state);
        return NULL;
    }

    if ((size_t)prob->constraint_matrix_num_nonzeros < state.coo_matrix.nnz)
    {
        fprintf(stderr, "WARNING: %zu duplicate matrix entries were summed.\n",
                state.coo_matrix.nnz -
                    (size_t)prob->constraint_matrix_num_nonzeros);
    }
    if (state.warn_unknown_row > MPS_WARN_PRINT_LIMIT)
    {
        fprintf(stderr,
                "WARNING: %d entries with unknown row names were ignored "
                "in total.\n", state.warn_unknown_row);
    }
    if (state.warn_unknown_col > MPS_WARN_PRINT_LIMIT)
    {
        fprintf(stderr,
                "WARNING: %d columns appeared only in BOUNDS in total.\n",
                state.warn_unknown_col);
    }
    if (state.warn_bad_number > MPS_WARN_PRINT_LIMIT)
    {
        fprintf(stderr,
                "WARNING: %d invalid numeric fields were treated as 0 "
                "in total.\n", state.warn_bad_number);
    }

    free_parser_state(&state);
    return prob;
}

static int parse_rows_section(MpsParserState *state, char **tokens,
                              int n_tokens)
{
    if (n_tokens < 2)
        return 0;

    if (n_tokens > 2)
    {
        fprintf(stderr,
                "ERROR: Row name '%s ...' contains spaces; "
                "fixed-format MPS is not supported.\n",
                tokens[1]);
        return -1;
    }

    if (tokens[0][1] != '\0' || strchr("NELG", tokens[0][0]) == NULL)
    {
        fprintf(stderr, "ERROR: Invalid row type '%s' for row '%s'.\n",
                tokens[0], tokens[1]);
        return -1;
    }

    if (state->num_buffered_rows >= state->buffered_rows_capacity)
    {
        state->buffered_rows_capacity = (state->buffered_rows_capacity == 0)
                                            ? 64
                                            : state->buffered_rows_capacity * 2;
        state->buffered_rows = (BufferedRow *)safe_realloc(
            state->buffered_rows,
            state->buffered_rows_capacity * sizeof(BufferedRow));
    }

    BufferedRow *new_row = &state->buffered_rows[state->num_buffered_rows];
    new_row->type = tokens[0][0];
    new_row->name = strdup(tokens[1]);
    if (!new_row->name)
        return -1;

    state->num_buffered_rows++;
    return 0;
}

static int finalize_rows(MpsParserState *state)
{
    int obj_idx = -1;

    for (size_t i = 0; i < state->num_buffered_rows; ++i)
    {
        if (state->buffered_rows[i].type == 'N')
        {
            obj_idx = (int)i;
            break;
        }
    }

    if (obj_idx == -1)
    {
        fprintf(stderr,
                "WARNING: no objective (N) row found: using a zero objective.\n");
    }
    else
    {
        state->objective_row_name = strdup(state->buffered_rows[obj_idx].name);
        if (!state->objective_row_name)
            return -1;
    }

    for (size_t i = 0; i < state->num_buffered_rows; ++i)
    {
        if ((int)i == obj_idx)
            continue;

        char type = state->buffered_rows[i].type;
        if (type == 'E' || type == 'L' || type == 'G')
        {
            size_t current_size = state->row_map.size;
            if (current_size >= state->constraint_capacity)
            {
                state->constraint_capacity = (state->constraint_capacity == 0)
                                                 ? 64
                                                 : state->constraint_capacity * 2;
                state->constraint_types = (char *)safe_realloc(
                    state->constraint_types, state->constraint_capacity * sizeof(char));
            }
            namemap_put(&state->row_map, state->buffered_rows[i].name);
            state->constraint_types[current_size] = type;
        }
        else
        {
            namemap_put(&state->free_row_map, state->buffered_rows[i].name);
        }
    }
    size_t num_constraints = state->row_map.size;
    if (num_constraints > 0)
    {
        state->constraint_lower_bounds =
            safe_malloc(num_constraints * sizeof(double));
        state->constraint_upper_bounds =
            safe_malloc(num_constraints * sizeof(double));

        for (size_t i = 0; i < num_constraints; ++i)
        {
            char type = state->constraint_types[i];
            if (type == 'L')
            {
                state->constraint_lower_bounds[i] = -INFINITY;
                state->constraint_upper_bounds[i] = 0.0;
            }
            else if (type == 'G')
            {
                state->constraint_lower_bounds[i] = 0.0;
                state->constraint_upper_bounds[i] = INFINITY;
            }
            else // 'E'
            {
                state->constraint_lower_bounds[i] = 0.0;
                state->constraint_upper_bounds[i] = 0.0;
            }
        }
    }
    return 0;
}

static int parse_columns_section(MpsParserState *state, char **tokens,
                                 int n_tokens)
{
    if (n_tokens < 2)
        return 0;

    if (n_tokens >= 2 && strcmp(tokens[1], "'MARKER'") == 0)
    {
        if (n_tokens >= 3 && strcmp(tokens[2], "'INTORG'") == 0 &&
            state->warn_integrality == 0)
        {
            state->warn_integrality = 1;
            fprintf(stderr,
                    "WARNING: integrality markers ignored: "
                    "solving the LP relaxation.\n");
        }
        return 0;
    }

    const char *col_name = NULL;
    int pair_start_index;

    if (n_tokens % 2 != 0)
    {
        free(state->current_col_name);
        state->current_col_name = strdup(tokens[0]);
        if (!state->current_col_name)
            return -1;

        col_name = state->current_col_name;
        pair_start_index = 1;
    }
    else
    {
        if (!state->current_col_name)
        {
            fprintf(stderr,
                    "ERROR: Column data found before any column name was defined.\n");
            return -1;
        }
        col_name = state->current_col_name;
        pair_start_index = 0;
    }

    if (!ensure_column_capacity(state))
        return -1;

    int col_idx = namemap_put(&state->col_map, col_name);
    if (col_idx == -1)
        return -1;

    for (int i = pair_start_index; i + 1 < n_tokens; i += 2)
    {
        const char *row_name = tokens[i];
        double value = parse_mps_number(state, tokens[i + 1]);

        if (state->objective_row_name &&
            strcmp(row_name, state->objective_row_name) == 0)
        {
            state->objective_coeffs[col_idx] += value;
        }
        else
        {
            int row_idx = namemap_get(&state->row_map, row_name);
            if (row_idx != -1)
            {
                if (add_coo_entry(&state->coo_matrix, row_idx, col_idx, value) != 0)
                {
                    return -1;
                }
            }
            else if (namemap_get(&state->free_row_map, row_name) == -1)
            {
                parser_warning(&state->warn_unknown_row,
                               "unknown row '%s' in COLUMNS section: "
                               "entry ignored.\n", row_name);
            }
        }
    }
    return 0;
}

static bool is_known_row(const MpsParserState *state, const char *name)
{
    if (namemap_get(&state->row_map, name) != -1)
        return true;
    if (namemap_get(&state->free_row_map, name) != -1)
        return true;
    return state->objective_row_name &&
           strcmp(name, state->objective_row_name) == 0;
}

static int parse_rhs_section(MpsParserState *state, char **tokens,
                             int n_tokens)
{
    for (int i = is_known_row(state, tokens[0]) ? 0 : 1; i + 1 < n_tokens; i += 2)
    {
        const char *row_name = tokens[i];
        double value = parse_mps_number(state, tokens[i + 1]);

        if (state->objective_row_name &&
            strcmp(row_name, state->objective_row_name) == 0)
        {
            state->objective_constant = -value;
        }
        else
        {
            int row_idx = namemap_get(&state->row_map, row_name);
            if (row_idx != -1)
            {
                char type = state->constraint_types[row_idx];
                if (type == 'L')
                    state->constraint_upper_bounds[row_idx] = value;
                else if (type == 'G')
                    state->constraint_lower_bounds[row_idx] = value;
                else
                {
                    state->constraint_lower_bounds[row_idx] = value;
                    state->constraint_upper_bounds[row_idx] = value;
                }
            }
            else if (namemap_get(&state->free_row_map, row_name) == -1)
            {
                parser_warning(&state->warn_unknown_row,
                               "unknown row '%s' in RHS section: "
                               "entry ignored.\n", row_name);
            }
        }
    }
    return 0;
}

static int parse_ranges_section(MpsParserState *state, char **tokens,
                                int n_tokens)
{
    for (int i = is_known_row(state, tokens[0]) ? 0 : 1; i + 1 < n_tokens; i += 2)
    {
        const char *row_name = tokens[i];
        double range_val = parse_mps_number(state, tokens[i + 1]);
        int row_idx = namemap_get(&state->row_map, row_name);

        if (row_idx == -1 && !is_known_row(state, row_name))
        {
            parser_warning(&state->warn_unknown_row,
                           "unknown row '%s' in RANGES section: "
                           "entry ignored.\n", row_name);
        }

        if (row_idx != -1)
        {
            char type = state->constraint_types[row_idx];
            double rhs = (type == 'L') ? state->constraint_upper_bounds[row_idx]
                                       : state->constraint_lower_bounds[row_idx];

            if (type == 'G')
            {
                state->constraint_upper_bounds[row_idx] = rhs + fabs(range_val);
            }
            else if (type == 'L')
            {
                state->constraint_lower_bounds[row_idx] = rhs - fabs(range_val);
            }
            else if (type == 'E')
            {
                if (range_val >= 0)
                {
                    state->constraint_upper_bounds[row_idx] = rhs + range_val;
                }
                else
                {
                    state->constraint_lower_bounds[row_idx] = rhs + range_val;
                }
            }
        }
    }
    return 0;
}

static int parse_bounds_section(MpsParserState *state, char **tokens,
                                int n_tokens)
{
    if (n_tokens < 2)
        return 0;

    const char *bound_type = tokens[0];
    bool has_value = strcmp(bound_type, "LO") == 0 || strcmp(bound_type, "LI") == 0 ||
                     strcmp(bound_type, "UP") == 0 || strcmp(bound_type, "UI") == 0 ||
                     strcmp(bound_type, "FX") == 0;
    bool no_value = strcmp(bound_type, "FR") == 0 || strcmp(bound_type, "MI") == 0 ||
                    strcmp(bound_type, "PL") == 0 || strcmp(bound_type, "BV") == 0;
    if (!has_value && !no_value)
    {
        fprintf(stderr, "ERROR: Unknown bound type '%s'.\n", bound_type);
        return -1;
    }

    const char *col_name;
    const char *value_token = NULL;
    if (namemap_get(&state->col_map, tokens[1]) != -1 || n_tokens == 2)
    {
        col_name = tokens[1];
        if (n_tokens >= 3)
            value_token = tokens[2];
    }
    else
    {
        col_name = tokens[2];
        if (n_tokens >= 4)
            value_token = tokens[3];
    }

    if (has_value && value_token == NULL)
    {
        fprintf(stderr,
                "ERROR: Malformed %s bound line (missing value or unknown "
                "column '%s').\n", bound_type, col_name);
        return -1;
    }
    double value = has_value ? parse_mps_number(state, value_token) : 0.0;

    int col_idx = namemap_get(&state->col_map, col_name);
    if (col_idx == -1)
    {
        if (!ensure_column_capacity(state))
            return -1;
        col_idx = namemap_put(&state->col_map, col_name);
        if (col_idx == -1)
            return -1;
        parser_warning(&state->warn_unknown_col,
                       "column '%s' appears only in BOUNDS: "
                       "created with default bounds.\n", col_name);
    }

    if (strcmp(bound_type, "LO") == 0 || strcmp(bound_type, "LI") == 0)
    {
        state->var_lower_bounds[col_idx] = value;
    }
    else if (strcmp(bound_type, "UP") == 0 || strcmp(bound_type, "UI") == 0)
    {
        state->var_upper_bounds[col_idx] = value;
        // MPS convention: a negative upper bound on a variable whose lower bound
        // is still the default 0 implies a lower bound of -infinity (otherwise the
        // interval [0, value] would be empty). LI/MI set before this leave a
        // non-zero/-inf lower, so they are not overridden.
        if (value < 0.0 && state->var_lower_bounds[col_idx] == 0.0)
        {
            state->var_lower_bounds[col_idx] = -INFINITY;
        }
    }
    else if (strcmp(bound_type, "FX") == 0)
    {
        state->var_lower_bounds[col_idx] = value;
        state->var_upper_bounds[col_idx] = value;
    }
    else if (strcmp(bound_type, "FR") == 0)
    {
        state->var_lower_bounds[col_idx] = -INFINITY;
        state->var_upper_bounds[col_idx] = INFINITY;
    }
    else if (strcmp(bound_type, "MI") == 0)
    {
        state->var_lower_bounds[col_idx] = -INFINITY;
    }
    else if (strcmp(bound_type, "PL") == 0)
    {
        state->var_upper_bounds[col_idx] = INFINITY;
    }
    else if (strcmp(bound_type, "BV") == 0)
    {
        state->var_lower_bounds[col_idx] = 0.0;
        state->var_upper_bounds[col_idx] = 1.0;
    }
    return 0;
}

static void free_parser_state(MpsParserState *state)
{
    if (!state)
        return;

    namemap_free(&state->row_map);
    namemap_free(&state->col_map);
    namemap_free(&state->free_row_map);

    if (state->buffered_rows)
    {
        for (size_t i = 0; i < state->num_buffered_rows; ++i)
        {
            free(state->buffered_rows[i].name);
        }
        free(state->buffered_rows);
    }

    free(state->coo_matrix.row_indices);
    free(state->coo_matrix.col_indices);
    free(state->coo_matrix.values);

    free(state->constraint_types);
    free(state->objective_coeffs);
    free(state->var_lower_bounds);
    free(state->var_upper_bounds);
    free(state->constraint_lower_bounds);
    free(state->constraint_upper_bounds);
    free(state->objective_row_name);
    free(state->current_col_name);
}
