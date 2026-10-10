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
#include "hpdlp.h"
#include "error_handling.h"
#include "matrix_formats.h"
#include "utils.h"
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define READER_BUFFER_SIZE (4 * 1024 * 1024)
#define MPS_MAX_LINE_LENGTH 4096
#define MPS_MAX_NUMBER_LENGTH 64
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

typedef struct
{
    gzFile file;
    char *buffer;
    char *current_pos;
    char *end_pos;
    bool read_error;
} FastLineReader;

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
    NameMap row_map;
    NameMap col_map;
    NameMap free_row_map;

    CooMatrix coo_matrix;
    BufferedRow *buffered_rows;
    size_t num_buffered_rows;
    size_t buffered_rows_capacity;

    char *constraint_types;
    double *constraint_rhs;
    double *constraint_ranges;
    double *objective_coeffs;
    double *var_lower_bounds;
    double *var_upper_bounds;
    bool *col_binary_default;
    bool *col_has_lower;

    size_t col_capacity;

    char *objective_row_name;
    char *requested_objective_row_name;
    char *current_col_name;
    int current_col_idx;
    bool in_integer_block;
    double objective_constant;
    objective_sense_t objective_sense;
    bool has_error;

    int num_unknown_row_entries;
    int num_unknown_cols;
    bool warned_integrality;
    bool warned_sos;
} MpsParserState;

typedef enum
{
    SEC_NONE,
    SEC_ROWS,
    SEC_COLUMNS,
    SEC_RHS,
    SEC_RANGES,
    SEC_BOUNDS,
    SEC_OBJSENSE,
    SEC_OBJNAME,
    SEC_SOS,
    SEC_UNSUPPORTED,
    SEC_ENDATA
} MpsSection;

static const char *const MPS_UNSUPPORTED_SECTIONS[] = {
    "QUADOBJ", "QMATRIX", "QSECTION", "QCMATRIX", "CSECTION",
    "INDICATORS", "DELAYEDROWS", "MODELCUTS", "USERCUTS", "GENCONS",
    "PWLOBJ", "PWLNAM", "PWLCON"};

static lp_problem_t *parse_mps_file(const char *filename);
static FastLineReader *fast_reader_open(const char *filename);
static void name_map_init(NameMap *map, size_t num_buckets);
static int parse_mps_stream(MpsParserState *state, FastLineReader *reader);
static char *fast_reader_gets(FastLineReader *reader, char *line_buf,
                             int line_buf_size);
static bool fast_reader_refill(FastLineReader *reader);
static int tokenize_mps_line(char *line, char **tokens);
static MpsSection identify_section(char **tokens, int num_tokens);
static bool token_has_prefix_ci(const char *token, const char *prefix);
static int finalize_rows(MpsParserState *state);
static char *copy_name(const char *name);
static int register_row(MpsParserState *state, NameMap *map, const char *name);
static int name_map_get(const NameMap *map, const char *name);
static unsigned long hash_string(const char *str);
static int name_map_put(NameMap *map, const char *name);
static void name_map_resize(NameMap *map);
static void free_buffered_rows(MpsParserState *state);
static int parse_section_data(MpsParserState *state, MpsSection section,
                              char **tokens, int num_tokens);
static int parse_rows_section(MpsParserState *state, char **tokens,
                              int num_tokens);
static int parse_columns_section(MpsParserState *state, char **tokens,
                                 int num_tokens);
static int select_column(MpsParserState *state, const char *name);
static void ensure_column_capacity(MpsParserState *state);
static double parse_mps_number(MpsParserState *state, const char *token);
static int find_constraint_row(MpsParserState *state, const char *name,
                               const char *section_name);
static void parser_warning(int *count, const char *fmt, ...);
static void add_coo_entry(CooMatrix *coo, int row, int col, double value);
static int parse_rhs_section(MpsParserState *state, char **tokens,
                             int num_tokens);
static int parse_ranges_section(MpsParserState *state, char **tokens,
                                int num_tokens);
static int parse_bounds_section(MpsParserState *state, char **tokens,
                                int num_tokens);
static void fast_reader_close(FastLineReader *reader);
static lp_problem_t *build_lp_problem(MpsParserState *state);
static int compute_constraint_bounds(const MpsParserState *state,
                                     double *lower, double *upper);
static void print_parser_warnings(const MpsParserState *state,
                                  const lp_problem_t *prob);
static void free_parser_state(MpsParserState *state);
static void name_map_free(NameMap *map);

lp_problem_t *read_mps_file(const char *filename)
{
    error_scope_t scope;
    enter_error_scope(&scope);
    if (setjmp(scope.env) != 0)
    {
        leave_error_scope();
        return NULL;
    }
    lp_problem_t *prob = parse_mps_file(filename);
    leave_error_scope();
    return prob;
}

static lp_problem_t *parse_mps_file(const char *filename)
{
    FastLineReader *reader = fast_reader_open(filename);
    if (!reader)
    {
        report_error("could not open file '%s'", filename);
        return NULL;
    }

    MpsParserState state = {0};
    name_map_init(&state.row_map, 1024);
    name_map_init(&state.col_map, 1024);
    name_map_init(&state.free_row_map, 16);

    int status = parse_mps_stream(&state, reader);
    fast_reader_close(reader);

    lp_problem_t *prob = NULL;
    if (status == 0)
        prob = build_lp_problem(&state);

    if (prob)
        print_parser_warnings(&state, prob);
    free_parser_state(&state);
    return prob;
}

static FastLineReader *fast_reader_open(const char *filename)
{
    gzFile file = gzopen(filename, "rb");
    if (!file)
        return NULL;

    FastLineReader *reader = safe_calloc(1, sizeof(FastLineReader));
    reader->file = file;
    reader->buffer = safe_malloc(READER_BUFFER_SIZE);
    reader->current_pos = reader->buffer;
    reader->end_pos = reader->buffer;
    return reader;
}

static void name_map_init(NameMap *map, size_t num_buckets)
{
    map->num_buckets = num_buckets;
    map->size = 0;
    map->buckets = safe_calloc(num_buckets, sizeof(NameNode *));
}

static int parse_mps_stream(MpsParserState *state, FastLineReader *reader)
{
    MpsSection section = SEC_NONE;
    bool rows_finalized = false;
    bool has_rows_section = false;
    bool has_columns_section = false;
    char line[MPS_MAX_LINE_LENGTH];

    while (fast_reader_gets(reader, line, sizeof(line)))
    {
        char *tokens[MPS_MAX_TOKENS];
        int num_tokens = tokenize_mps_line(line, tokens);
        if (num_tokens < 0)
            return -1;
        if (num_tokens == 0)
            continue;

        if (section == SEC_OBJNAME)
        {
            if (parse_section_data(state, section, tokens, num_tokens) != 0)
                return -1;
            section = SEC_NONE;
            continue;
        }

        if (section == SEC_NONE && strcmp(tokens[0], "NAME") == 0)
            continue;

        MpsSection next_section = identify_section(tokens, num_tokens);
        if (!rows_finalized && num_tokens == 2 && strcmp(tokens[0], "OBJNAME") == 0)
            next_section = SEC_OBJNAME;
        if (next_section == SEC_UNSUPPORTED)
        {
            report_error("MPS section '%s' is not supported", tokens[0]);
            return -1;
        }
        if (next_section != SEC_NONE)
        {
            if (next_section == SEC_OBJNAME && rows_finalized)
            {
                report_error("OBJNAME must precede column data");
                return -1;
            }
            if (next_section == SEC_ROWS && rows_finalized)
            {
                report_error("ROWS section must precede column data");
                return -1;
            }
            if (!rows_finalized &&
                (next_section == SEC_COLUMNS || next_section == SEC_RHS ||
                 next_section == SEC_RANGES || next_section == SEC_BOUNDS ||
                 next_section == SEC_ENDATA))
            {
                if (finalize_rows(state) != 0)
                    return -1;
                rows_finalized = true;
            }
            if (next_section == SEC_SOS && !state->warned_sos)
            {
                state->warned_sos = true;
                report_warning("SOS constraints ignored: solving the LP relaxation");
            }
            has_rows_section |= next_section == SEC_ROWS;
            has_columns_section |= next_section == SEC_COLUMNS;
            section = next_section;
            if (section == SEC_ENDATA)
                break;
            if ((section == SEC_OBJSENSE || section == SEC_OBJNAME) && num_tokens == 2)
            {
                if (parse_section_data(state, section, tokens + 1, 1) != 0)
                    return -1;
                section = SEC_NONE;
            }
            continue;
        }

        if (section == SEC_NONE)
        {
            report_error("data line starting with '%s' is outside "
                         "any MPS section", tokens[0]);
            return -1;
        }
        if (parse_section_data(state, section, tokens, num_tokens) != 0 ||
            state->has_error)
            return -1;
    }

    if (reader->read_error)
        return -1;
    if (section == SEC_OBJNAME)
    {
        report_error("OBJNAME section has no row name");
        return -1;
    }
    if (!has_rows_section || !has_columns_section)
    {
        report_error("MPS file has no %s section", has_rows_section ? "COLUMNS" : "ROWS");
        return -1;
    }
    return 0;
}

static char *fast_reader_gets(FastLineReader *reader, char *line_buf,
                             int line_buf_size)
{
    int len = 0;

    while (1)
    {
        if (reader->current_pos >= reader->end_pos && !fast_reader_refill(reader))
        {
            return !reader->read_error && len > 0 ? line_buf : NULL;
        }

        char *newline_pos = (char *)memchr(reader->current_pos, '\n',
                                          reader->end_pos - reader->current_pos);
        int bytes_to_copy = newline_pos
                                ? newline_pos - reader->current_pos + 1
                                : reader->end_pos - reader->current_pos;

        if (len + bytes_to_copy >= line_buf_size)
        {
            report_error("line too long (limit %d characters)", line_buf_size - 1);
            reader->read_error = true;
            return NULL;
        }

        memcpy(line_buf + len, reader->current_pos, bytes_to_copy);
        len += bytes_to_copy;
        reader->current_pos += bytes_to_copy;
        line_buf[len] = '\0';

        if (newline_pos)
        {
            return line_buf;
        }
    }
}

static bool fast_reader_refill(FastLineReader *reader)
{
    int bytes_read = gzread(reader->file, reader->buffer, READER_BUFFER_SIZE);
    if (bytes_read <= 0)
    {
        int error = Z_OK;
        gzerror(reader->file, &error);
        if (bytes_read < 0 || (error != Z_OK && error != Z_STREAM_END))
        {
            report_error("failed to read input file (corrupt or truncated file)");
            reader->read_error = true;
        }
        return false;
    }

    reader->current_pos = reader->buffer;
    reader->end_pos = reader->buffer + bytes_read;
    return true;
}

static int tokenize_mps_line(char *line, char **tokens)
{
    if (line[0] == '*' || line[0] == '\n' || line[0] == '\r')
        return 0;

    int count = 0;
    char *saveptr;
    for (char *token = strtok_r(line, " \t\n\r", &saveptr);
         token != NULL && token[0] != '$';
         token = strtok_r(NULL, " \t\n\r", &saveptr))
    {
        if (count == MPS_MAX_TOKENS)
        {
            report_error("too many fields on one line (limit %d)", MPS_MAX_TOKENS);
            return -1;
        }
        tokens[count++] = token;
    }

    if (count > 0 && tokens[0][0] == '*')
    {
        size_t length = strlen(tokens[0]);
        if (length == 1 || tokens[0][length - 1] != '*')
            return 0;
    }
    return count;
}

static MpsSection identify_section(char **tokens, int num_tokens)
{
    const char *name = tokens[0];
    bool is_objsense = strcmp(name, "OBJSENSE") == 0 ||
                       strcmp(name, "OBJSENS") == 0;
    bool inline_sense = is_objsense && num_tokens == 2 &&
                        (token_has_prefix_ci(tokens[1], "MIN") ||
                         token_has_prefix_ci(tokens[1], "MAX"));
    if (num_tokens != 1 && !inline_sense)
        return SEC_NONE;

    if (is_objsense)
        return SEC_OBJSENSE;

    static const struct
    {
        const char *name;
        MpsSection section;
    } sections[] = {
        {"ROWS", SEC_ROWS},
        {"OBJNAME", SEC_OBJNAME},
        {"COLUMNS", SEC_COLUMNS},
        {"RHS", SEC_RHS},
        {"RANGES", SEC_RANGES},
        {"BOUNDS", SEC_BOUNDS},
        {"SOS", SEC_SOS},
        {"SETS", SEC_SOS},
        {"ENDATA", SEC_ENDATA},
    };

    for (size_t i = 0; i < sizeof(sections) / sizeof(sections[0]); ++i)
    {
        if (strcmp(name, sections[i].name) == 0)
            return sections[i].section;
    }
    for (size_t i = 0; i < sizeof(MPS_UNSUPPORTED_SECTIONS) /
                               sizeof(MPS_UNSUPPORTED_SECTIONS[0]); ++i)
    {
        if (strcmp(name, MPS_UNSUPPORTED_SECTIONS[i]) == 0)
            return SEC_UNSUPPORTED;
    }
    return SEC_NONE;
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

static int finalize_rows(MpsParserState *state)
{
    int objective_row_idx = -1;

    for (size_t i = 0; i < state->num_buffered_rows; ++i)
    {
        if (state->buffered_rows[i].type == 'N' &&
            (!state->requested_objective_row_name ||
             strcmp(state->buffered_rows[i].name, state->requested_objective_row_name) == 0))
        {
            objective_row_idx = (int)i;
            break;
        }
    }

    if (objective_row_idx == -1)
    {
        if (state->requested_objective_row_name)
        {
            report_error("OBJNAME '%s' does not name an N row",
                         state->requested_objective_row_name);
            return -1;
        }
        report_warning("no objective (N) row found: using a zero objective");
    }
    else
    {
        state->objective_row_name =
            copy_name(state->buffered_rows[objective_row_idx].name);
    }

    state->constraint_types = safe_malloc(state->num_buffered_rows * sizeof(char));
    for (size_t i = 0; i < state->num_buffered_rows; ++i)
    {
        const BufferedRow *row = &state->buffered_rows[i];
        bool is_constraint = row->type != 'N';
        NameMap *map = is_constraint ? &state->row_map : &state->free_row_map;
        if (register_row(state, map, row->name) != 0)
            return -1;
        if (is_constraint)
            state->constraint_types[state->row_map.size - 1] = row->type;
    }
    free_buffered_rows(state);

    size_t num_constraints = state->row_map.size;
    state->constraint_rhs = safe_malloc(num_constraints * sizeof(double));
    state->constraint_ranges = safe_malloc(num_constraints * sizeof(double));
    for (size_t i = 0; i < num_constraints; ++i)
    {
        state->constraint_rhs[i] = 0.0;
        state->constraint_ranges[i] = NAN;
    }
    return 0;
}

static char *copy_name(const char *name)
{
    size_t length = strlen(name) + 1;
    char *copy = safe_malloc(length);
    memcpy(copy, name, length);
    return copy;
}

static int register_row(MpsParserState *state, NameMap *map, const char *name)
{
    if (name_map_get(&state->row_map, name) != -1 ||
        name_map_get(&state->free_row_map, name) != -1)
    {
        report_error("row '%s' is defined more than once", name);
        return -1;
    }
    return name_map_put(map, name) == -1 ? -1 : 0;
}

static int name_map_get(const NameMap *map, const char *name)
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

static int name_map_put(NameMap *map, const char *name)
{
    int index = name_map_get(map, name);
    if (index != -1)
        return index;

    if (map->size >= INT_MAX)
    {
        report_error("too many rows or columns (limit %d)", INT_MAX);
        return -1;
    }
    if (map->size >= map->num_buckets * 0.75)
    {
        name_map_resize(map);
    }

    unsigned long h = hash_string(name) % (unsigned long)map->num_buckets;
    NameNode *new_node = safe_malloc(sizeof(NameNode));
    new_node->name = copy_name(name);
    new_node->index = (int)map->size++;
    new_node->next = map->buckets[h];
    map->buckets[h] = new_node;

    return new_node->index;
}

static void name_map_resize(NameMap *map)
{
    size_t old_num_buckets = map->num_buckets;
    NameNode **old_buckets = map->buckets;

    size_t new_num_buckets = old_num_buckets * 2;
    map->num_buckets = new_num_buckets;
    map->buckets = safe_calloc(new_num_buckets, sizeof(NameNode *));

    for (size_t i = 0; i < old_num_buckets; ++i)
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

static void free_buffered_rows(MpsParserState *state)
{
    for (size_t i = 0; i < state->num_buffered_rows; ++i)
    {
        free(state->buffered_rows[i].name);
    }
    free(state->buffered_rows);
    state->buffered_rows = NULL;
    state->num_buffered_rows = 0;
    state->buffered_rows_capacity = 0;
}

static int parse_section_data(MpsParserState *state, MpsSection section,
                              char **tokens, int num_tokens)
{
    switch (section)
    {
    case SEC_ROWS:
        return parse_rows_section(state, tokens, num_tokens);
    case SEC_COLUMNS:
        return parse_columns_section(state, tokens, num_tokens);
    case SEC_RHS:
        return parse_rhs_section(state, tokens, num_tokens);
    case SEC_RANGES:
        return parse_ranges_section(state, tokens, num_tokens);
    case SEC_BOUNDS:
        return parse_bounds_section(state, tokens, num_tokens);
    case SEC_OBJNAME:
        if (num_tokens != 1)
        {
            report_error("OBJNAME data line must hold exactly one row name");
            return -1;
        }
        free(state->requested_objective_row_name);
        state->requested_objective_row_name = copy_name(tokens[0]);
        return 0;
    case SEC_OBJSENSE:
        if (token_has_prefix_ci(tokens[0], "MAX"))
        {
            state->objective_sense = OBJECTIVE_SENSE_MAXIMIZE;
        }
        else if (!token_has_prefix_ci(tokens[0], "MIN"))
        {
            report_warning("unrecognized OBJSENSE value '%s' ignored "
                           "(minimize assumed)", tokens[0]);
        }
        return 0;
    default:
        return 0;
    }
}

static int parse_rows_section(MpsParserState *state, char **tokens,
                              int num_tokens)
{
    if (num_tokens < 2)
    {
        report_error("ROWS line '%s' needs a row type and a name", tokens[0]);
        return -1;
    }

    if (num_tokens > 2)
    {
        report_error("row name '%s ...' contains spaces; "
                     "fixed-format MPS is not supported",
                     tokens[1]);
        return -1;
    }

    if (tokens[0][1] != '\0' || strchr("NELG", tokens[0][0]) == NULL)
    {
        report_error("invalid row type '%s' for row '%s'", tokens[0], tokens[1]);
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
    new_row->name = copy_name(tokens[1]);
    state->num_buffered_rows++;
    return 0;
}

static int parse_columns_section(MpsParserState *state, char **tokens,
                                 int num_tokens)
{
    if (num_tokens >= 2 && strcmp(tokens[1], "'MARKER'") == 0)
    {
        if (num_tokens >= 3 && strcmp(tokens[2], "'INTORG'") == 0)
        {
            state->in_integer_block = true;
            if (!state->warned_integrality)
            {
                state->warned_integrality = true;
                report_warning("integrality markers ignored: solving the LP relaxation");
            }
        }
        else if (num_tokens >= 3 && strcmp(tokens[2], "'INTEND'") == 0)
        {
            state->in_integer_block = false;
        }
        return 0;
    }

    int pair_start_index = num_tokens % 2;
    if (pair_start_index == 1)
    {
        if (select_column(state, tokens[0]) != 0)
            return -1;
    }
    else if (!state->current_col_name)
    {
        report_error("column data found before any column name was defined");
        return -1;
    }

    int col_idx = state->current_col_idx;

    for (int i = pair_start_index; i + 1 < num_tokens; i += 2)
    {
        const char *row_name = tokens[i];
        double value = parse_mps_number(state, tokens[i + 1]);

        if (state->objective_row_name &&
            strcmp(row_name, state->objective_row_name) == 0)
        {
            state->objective_coeffs[col_idx] += value;
            continue;
        }

        int row_idx = find_constraint_row(state, row_name, "COLUMNS");
        if (row_idx != -1)
            add_coo_entry(&state->coo_matrix, row_idx, col_idx, value);
    }
    return 0;
}

static int select_column(MpsParserState *state, const char *name)
{
    if (state->current_col_name && strcmp(name, state->current_col_name) == 0)
        return 0;

    free(state->current_col_name);
    state->current_col_name = copy_name(name);
    ensure_column_capacity(state);

    size_t previous_size = state->col_map.size;
    state->current_col_idx = name_map_put(&state->col_map, name);
    if (state->current_col_idx == -1)
        return -1;
    if (state->col_map.size > previous_size && state->in_integer_block)
        state->col_binary_default[state->current_col_idx] = true;
    return 0;
}

static void ensure_column_capacity(MpsParserState *state)
{
    if (state->col_map.size < state->col_capacity)
    {
        return;
    }

    size_t new_capacity = (state->col_capacity == 0) ? 256 : state->col_capacity * 2;

    state->objective_coeffs =
        (double *)safe_realloc(state->objective_coeffs, new_capacity * sizeof(double));
    state->var_lower_bounds =
        (double *)safe_realloc(state->var_lower_bounds, new_capacity * sizeof(double));
    state->var_upper_bounds =
        (double *)safe_realloc(state->var_upper_bounds, new_capacity * sizeof(double));
    state->col_binary_default = (bool *)safe_realloc(
        state->col_binary_default, new_capacity * sizeof(bool));
    state->col_has_lower = (bool *)safe_realloc(
        state->col_has_lower, new_capacity * sizeof(bool));

    for (size_t i = state->col_capacity; i < new_capacity; ++i)
    {
        state->objective_coeffs[i] = 0.0;
        state->var_lower_bounds[i] = 0.0;
        state->var_upper_bounds[i] = INFINITY;
        state->col_binary_default[i] = false;
        state->col_has_lower[i] = false;
    }

    state->col_capacity = new_capacity;
}

static double parse_mps_number(MpsParserState *state, const char *token)
{
    char buf[MPS_MAX_NUMBER_LENGTH];
    size_t len = strlen(token);
    double value = NAN;
    if (len < sizeof(buf) && strpbrk(token, "xX") == NULL)
    {
        memcpy(buf, token, len + 1);
        char *exponent = strpbrk(buf, "Dd");
        if (exponent)
            *exponent = 'E';
        char *endptr;
        value = strtod(buf, &endptr);
        if (endptr == buf || *endptr != '\0')
            value = NAN;
    }
    if (!isfinite(value))
    {
        report_error("numeric field '%s' is not a finite number", token);
        state->has_error = true;
        return 0.0;
    }
    return value;
}

static int find_constraint_row(MpsParserState *state, const char *name,
                               const char *section_name)
{
    int row_idx = name_map_get(&state->row_map, name);
    if (row_idx == -1 && name_map_get(&state->free_row_map, name) == -1)
    {
        parser_warning(&state->num_unknown_row_entries,
                       "unknown row '%s' in %s section: entry ignored",
                       name, section_name);
    }
    return row_idx;
}

static void parser_warning(int *count, const char *fmt, ...)
{
    ++*count;
    if (*count > MPS_WARN_PRINT_LIMIT)
        return;
    char message[MPS_MAX_LINE_LENGTH];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);
    report_warning("%s", message);
    if (*count == MPS_WARN_PRINT_LIMIT)
        report_warning("further warnings of this kind suppressed");
}

static void add_coo_entry(CooMatrix *coo, int row, int col, double value)
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
}

static int parse_rhs_section(MpsParserState *state, char **tokens,
                             int num_tokens)
{
    if (num_tokens < 2)
    {
        report_error("RHS line '%s' has no row/value pair", tokens[0]);
        return -1;
    }

    for (int i = num_tokens % 2; i + 1 < num_tokens; i += 2)
    {
        const char *row_name = tokens[i];
        double value = parse_mps_number(state, tokens[i + 1]);

        if (state->objective_row_name &&
            strcmp(row_name, state->objective_row_name) == 0)
        {
            state->objective_constant = -value;
            continue;
        }

        int row_idx = find_constraint_row(state, row_name, "RHS");
        if (row_idx != -1)
            state->constraint_rhs[row_idx] = value;
    }
    return 0;
}

static int parse_ranges_section(MpsParserState *state, char **tokens,
                                int num_tokens)
{
    if (num_tokens < 2)
    {
        report_error("RANGES line '%s' has no row/value pair", tokens[0]);
        return -1;
    }

    for (int i = num_tokens % 2; i + 1 < num_tokens; i += 2)
    {
        const char *row_name = tokens[i];
        double range_val = parse_mps_number(state, tokens[i + 1]);
        int row_idx = find_constraint_row(state, row_name, "RANGES");
        if (row_idx != -1)
            state->constraint_ranges[row_idx] = range_val;
    }
    return 0;
}

static int parse_bounds_section(MpsParserState *state, char **tokens,
                                int num_tokens)
{
    const char *bound_type = tokens[0];
    bool has_value = strcmp(bound_type, "LO") == 0 || strcmp(bound_type, "LI") == 0 ||
                     strcmp(bound_type, "UP") == 0 || strcmp(bound_type, "UI") == 0 ||
                     strcmp(bound_type, "FX") == 0;
    bool no_value = strcmp(bound_type, "FR") == 0 || strcmp(bound_type, "MI") == 0 ||
                    strcmp(bound_type, "PL") == 0 || strcmp(bound_type, "BV") == 0;
    if (!has_value && !no_value)
    {
        report_error("unknown bound type '%s'", bound_type);
        return -1;
    }
    if (num_tokens < (has_value ? 3 : 2) || num_tokens > 4)
    {
        report_error("malformed %s bound line (%d fields)", bound_type, num_tokens);
        return -1;
    }

    int col_index = (num_tokens == 4 || (num_tokens == 3 && no_value)) ? 2 : 1;
    if (num_tokens == 3 && no_value &&
        name_map_get(&state->col_map, tokens[2]) == -1 &&
        name_map_get(&state->col_map, tokens[1]) != -1)
        col_index = 1;
    const char *col_name = tokens[col_index];
    double value = has_value ? parse_mps_number(state, tokens[col_index + 1]) : 0.0;

    int col_idx = name_map_get(&state->col_map, col_name);
    if (col_idx == -1)
    {
        ensure_column_capacity(state);
        col_idx = name_map_put(&state->col_map, col_name);
        if (col_idx == -1)
            return -1;
        parser_warning(&state->num_unknown_cols,
                       "column '%s' appears only in BOUNDS: "
                       "created with default bounds", col_name);
    }

    state->col_binary_default[col_idx] = false;

    if (strcmp(bound_type, "LO") == 0 || strcmp(bound_type, "LI") == 0)
    {
        state->var_lower_bounds[col_idx] = value;
        state->col_has_lower[col_idx] = true;
    }
    else if (strcmp(bound_type, "UP") == 0 || strcmp(bound_type, "UI") == 0)
    {
        state->var_upper_bounds[col_idx] = value;
        if (!state->col_has_lower[col_idx])
            state->var_lower_bounds[col_idx] = value < 0.0 ? -INFINITY : 0.0;
    }
    else if (strcmp(bound_type, "FX") == 0)
    {
        state->var_lower_bounds[col_idx] = value;
        state->var_upper_bounds[col_idx] = value;
        state->col_has_lower[col_idx] = true;
    }
    else if (strcmp(bound_type, "FR") == 0)
    {
        state->var_lower_bounds[col_idx] = -INFINITY;
        state->var_upper_bounds[col_idx] = INFINITY;
        state->col_has_lower[col_idx] = true;
    }
    else if (strcmp(bound_type, "MI") == 0)
    {
        state->var_lower_bounds[col_idx] = -INFINITY;
        state->col_has_lower[col_idx] = true;
    }
    else if (strcmp(bound_type, "PL") == 0)
    {
        state->var_upper_bounds[col_idx] = INFINITY;
    }
    else if (strcmp(bound_type, "BV") == 0)
    {
        state->var_lower_bounds[col_idx] = 0.0;
        state->var_upper_bounds[col_idx] = 1.0;
        state->col_has_lower[col_idx] = true;
    }
    return 0;
}

static void fast_reader_close(FastLineReader *reader)
{
    gzclose(reader->file);
    free(reader->buffer);
    free(reader);
}

static lp_problem_t *build_lp_problem(MpsParserState *state)
{
    if (state->coo_matrix.nnz > INT_MAX)
    {
        report_error("too many matrix entries (limit %d)", INT_MAX);
        return NULL;
    }

    for (size_t i = 0; i < state->col_map.size; ++i)
    {
        if (state->col_binary_default[i])
        {
            state->var_lower_bounds[i] = 0.0;
            state->var_upper_bounds[i] = 1.0;
        }
    }

    lp_problem_t *prob = safe_calloc(1, sizeof(lp_problem_t));

    prob->num_variables = (int)state->col_map.size;
    prob->num_constraints = (int)state->row_map.size;
    prob->objective_constant = state->objective_constant;
    prob->objective_sense = state->objective_sense;

    prob->objective_vector = state->objective_coeffs;
    prob->variable_lower_bound = state->var_lower_bounds;
    prob->variable_upper_bound = state->var_upper_bounds;
    state->objective_coeffs = NULL;
    state->var_lower_bounds = NULL;
    state->var_upper_bounds = NULL;

    prob->constraint_lower_bound =
        safe_malloc(state->row_map.size * sizeof(double));
    prob->constraint_upper_bound =
        safe_malloc(state->row_map.size * sizeof(double));
    if (compute_constraint_bounds(state, prob->constraint_lower_bound,
                                  prob->constraint_upper_bound) != 0)
    {
        lp_problem_free(prob);
        return NULL;
    }

    matrix_desc_t desc = {0};
    desc.m = prob->num_constraints;
    desc.n = prob->num_variables;
    desc.fmt = matrix_coo;
    desc.data.coo.nnz = (int)state->coo_matrix.nnz;
    desc.data.coo.row_ind = state->coo_matrix.row_indices;
    desc.data.coo.col_ind = state->coo_matrix.col_indices;
    desc.data.coo.vals = state->coo_matrix.values;

    matrix_format_status_t status = matrix_desc_to_csr(&desc,
                                                       &prob->constraint_matrix_row_pointers,
                                                       &prob->constraint_matrix_col_indices,
                                                       &prob->constraint_matrix_values,
                                                       &prob->constraint_matrix_num_nonzeros);
    if (status != MATRIX_FORMAT_OK)
    {
        report_error("failed to convert matrix to CSR format: %s",
                     matrix_format_status_message(status));
        lp_problem_free(prob);
        return NULL;
    }

    if (!validate_lp_problem_data(prob))
    {
        lp_problem_free(prob);
        return NULL;
    }

    return prob;
}

static int compute_constraint_bounds(const MpsParserState *state,
                                     double *lower, double *upper)
{
    for (size_t i = 0; i < state->row_map.size; ++i)
    {
        char type = state->constraint_types[i];
        double rhs = state->constraint_rhs[i];
        double range = state->constraint_ranges[i];
        lower[i] = type == 'L' ? -INFINITY : rhs;
        upper[i] = type == 'G' ? INFINITY : rhs;
        if (isnan(range))
            continue;

        if (type == 'G')
            upper[i] = rhs + fabs(range);
        else if (type == 'L')
            lower[i] = rhs - fabs(range);
        else if (range >= 0.0)
            upper[i] = rhs + range;
        else
            lower[i] = rhs + range;

        if (!isfinite(lower[i]) || !isfinite(upper[i]))
        {
            report_error("non-finite range bound for constraint %zu", i);
            return -1;
        }
    }
    return 0;
}

static void print_parser_warnings(const MpsParserState *state,
                                  const lp_problem_t *prob)
{
    if ((size_t)prob->constraint_matrix_num_nonzeros < state->coo_matrix.nnz)
    {
        report_warning("%zu duplicate matrix entries were summed",
                       state->coo_matrix.nnz -
                           (size_t)prob->constraint_matrix_num_nonzeros);
    }
    if (state->num_unknown_row_entries > MPS_WARN_PRINT_LIMIT)
    {
        report_warning("%d entries with unknown row names were ignored in total",
                       state->num_unknown_row_entries);
    }
    if (state->num_unknown_cols > MPS_WARN_PRINT_LIMIT)
    {
        report_warning("%d columns appeared only in BOUNDS in total",
                       state->num_unknown_cols);
    }
}

static void free_parser_state(MpsParserState *state)
{
    name_map_free(&state->row_map);
    name_map_free(&state->col_map);
    name_map_free(&state->free_row_map);
    free_buffered_rows(state);

    free(state->coo_matrix.row_indices);
    free(state->coo_matrix.col_indices);
    free(state->coo_matrix.values);

    free(state->constraint_types);
    free(state->constraint_rhs);
    free(state->constraint_ranges);
    free(state->objective_coeffs);
    free(state->var_lower_bounds);
    free(state->var_upper_bounds);
    free(state->col_binary_default);
    free(state->col_has_lower);
    free(state->objective_row_name);
    free(state->requested_objective_row_name);
    free(state->current_col_name);
}

static void name_map_free(NameMap *map)
{
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
}
