################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize
import copy
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS   = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')
##DEBUG
DEBUG_VARIABLE = False
error_Type = 0 #0 - no error, 1 wrong types example, 2 runtime, 3 syntax


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx+1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        # make limit value can work, cannot assume put 1 as a fake limit number
        if type(toks[idx-1]) != int:
            return idx, 1

        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#   BY ME
#   4. GROUPBY only has 1 column to group by :D. True for validation
#   5. What the fk happens with self joins?
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS   = ('join', 'on', 'as')

COMPARISONS = ('>=', '<=', '!=','=', '>', '<', '!')#hack ;)
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
AGG_OPS_NUMBER = ('count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

## We want a function going from SQL tree to simple SQL and then we can customize that function

def col_id_to_str(col_id):
    col_id=col_id[2:-2]
    if col_id == "all":
        col_id = "*"
    return col_id

def col_unit_str(col_unit):
    #col_unit: (agg_id, col_id, isDistinct(bool))
    agg_id = col_unit[0]
    col_id = col_id_to_str(col_unit[1])
    isDistinct = col_unit[2]
    no_aggr = "DISTINCT "*isDistinct + col_id
    ag_op = AGG_OPS[agg_id]
    #no aggregation 
    if ag_op == "none":
        return no_aggr
    else:
        return ag_op + "( " + no_aggr + " )"
    
def val_unit_str(val_unit):
    #val_unit: (unit_op, col_unit1, col_unit2) 
    unit_op = val_unit[0]
    col_unit1 = val_unit[1]
    col_unit2 = val_unit[2]
    
    if col_unit2 is None: #if thingy is none return boo boo
        return col_unit_str(col_unit1)
    else:
        return col_unit_str(col_unit1) + " " + UNIT_OPS[unit_op] + " " + col_unit_str(col_unit2)

def table_from_col_str(col_str):
    #given table.col1, return table. Used in FROM clause
    return col_str.split(".")[0]

def cond_unit_str(conds, from_clause = True):
    # cond_unit: (not_op, op_id, val_unit, val1, val2)
    # val1 can either be col_unit or it can be value (10.0). If it 
    return_me = ""
    for idx, cond in enumerate(conds):#The structure for joins is always table.col OP table.col (And never aggregation etc.)
        #condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
        if idx%2==0: #cond unit
            val_unit1 = cond[2]
            col_unit1 = val_unit1[1]
            not_op_id = cond[0]
            op_id = cond[1]
            col_unit2 = cond[3]
            val_2 = cond[4]
            #FROM t1 JOIN t2
            if from_clause:
                if idx == 0:
                    return_me += table_from_col_str(col_unit_str(col_unit1)) + " JOIN " + table_from_col_str(col_unit_str(col_unit2)) + " ON "
                else:
                    if table_from_col_str(col_unit_str(col_unit1)) in return_me: # if cond1 is in returnme
                        return_me += " JOIN " + table_from_col_str(col_unit_str(col_unit2)) + " ON "
                    else:#if cond2 is in returnme
                        return_me += " JOIN " + table_from_col_str(col_unit_str(col_unit1)) + " ON "
            #print(col_unit1, WHERE_OPS[op_id], col_unit2)
            not_op = ""
            if not_op_id:
                not_op = "NOT "
            
            if type(col_unit2) == type(()):#if col_unit2 is a tuple, then it is of type col_unit
                col_unit2 = col_unit_str(col_unit2)
            if type(col_unit2) == type(dict()):#if col_unit2 is a dictionary, parse SQL
                col_unit2 = intersect_except_union_to_SQL(col_unit2)
            if type(col_unit2) == type(1.0):#float
                col_unit2 = int(col_unit2) #we always assume integers... i think
            return_me += col_unit_str(col_unit1) + " " + not_op + WHERE_OPS[op_id] + " " + str(col_unit2) #t1.col1 OP t2.col2,
            if val_2:
                return_me += " AND "+str(val_2)#BETWEEN val1 and val2
        else:#cond = AND/OR. If it is from clause, then do not do an and.
            return_me += " "
            if not from_clause:
                return_me += cond+ " "
    return return_me

def select_to_SQL(sel):
    overall_distinct = sel[0]
    cols = sel[1]
    return_me = "SELECT "+"DISTINCT "*overall_distinct
    #'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
    for item in cols:
        #item = (agg_id, val_unit)
        #val_unit: (unit_op, col_unit1, col_unit2) -> Im assuming there is no unit_op or col_unit2.
        aggr_str = AGG_OPS[item[0]]
        val_unit = item[1] #(0, (0, '__singer.age__', False), None)) Im assuming that we never do stuff between two cols
        if aggr_str == "none":# (1, '__singer.age__', False) -> singer.agemax0
            return_me += val_unit_str(val_unit) + ", "
        else:
            return_me += aggr_str + "( " + val_unit_str(val_unit) + " ), "

    return return_me[:-2]

def from_to_SQL(fro):
    # val_unit: (unit_op, col_unit1, col_unit2)
    # table_unit: (table_type, col_unit/sql)
    # cond_unit: (not_op, op_id, val_unit, val1, val2)

    table_units = fro["table_units"] 
    conds = fro["conds"]#we only need the conds to construct SQL
    return_me = "FROM "

    s=False
    for table_unit in table_units:
        #handle the special case when table unit is an SQL query!
        if table_unit[0]=="sql":#if the thingy is an sql query
            return_me += intersect_except_union_to_SQL(table_unit[1])
            s=True
    if s:
        return return_me
    if conds:#there will be joins if conditions exist
        return_me += cond_unit_str(conds) #This ignores JOIN
    
    else: # SIMPLE QUERY, use table_unit: (table_type, col_unit/sql)
        for table_unit in table_units:

            #table_unit = table_units[0]
            return_me += col_id_to_str(table_unit[1]) +", "
        return_me = return_me[:-2]
    return return_me

def where_to_SQL(whe):
    #condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
    return_me = "WHERE "
    if whe:
        return_me+=cond_unit_str(whe, from_clause=False)
    else:
        return_me+="NONE"
    return return_me

def groupBy_to_SQL(gro):
    #   'groupBy': [col_unit1, col_unit2, ...]
    #if len(gro) > 1: # Heh always only single group by column
    #    print("GREATER THAN 1!")
    #    breakpoint()
    return_me = "GROUP BY "
    
    if gro:#group by not empty.
        for group in gro:
            return_me += col_unit_str(gro[0]) + ", "
        return_me = return_me[:-2]
        return return_me
    else:
        return "GROUP BY NONE"
    
def orderBy_to_SQL(ord):
    #'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
    return_me = ""
    if ord:#ord by not empty.
        return_me = "ORDER BY "
        for val_unit in ord[1]:
            return_me += val_unit_str(val_unit) + ", "
        return return_me[:-2] + " " + ord[0].upper()
    else:
        return "ORDER BY NONE"

def limit_to_SQL(lim):
    #   'limit': None/limit value
    if lim:
        return "LIMIT "+str(lim)
    else:
        return "LIMIT NONE"


def having_to_SQL(hav):
    #condition: [cond_unit1, 'and'/'or', cond_unit2, ...] (same as where)
    return_me = "HAVING "
    if hav:
        return_me+=cond_unit_str(hav, from_clause=False)
    else:
        return_me+="NONE"
    return return_me


def intersect_except_union_to_SQL(int, prefix="", brackets=True):
    #   'intersect': None/sql
    #   'except': None/sql
    #   'union': None/sql
    if int:
        return prefix+" "+"("*brackets + sql_tree_to_SimpleSQL(int) + ")"*brackets
    else:
        return prefix+" NONE"

#convert AST to simple SQL
def sql_tree_to_SimpleSQL(query):
    select = select_to_SQL(query["select"])
    if DEBUG_VARIABLE:
        print("select: ", select)
    from_clause = from_to_SQL(query["from"])
    if DEBUG_VARIABLE:
        print("from: ", from_clause)
    where_clause = where_to_SQL(query["where"])
    if DEBUG_VARIABLE:
        print("where: ", where_clause)
    groupBy_clause = groupBy_to_SQL(query["groupBy"])
    if DEBUG_VARIABLE:
        print("groupby: ", groupBy_clause)
    orderBy_clause = orderBy_to_SQL(query["orderBy"])
    if DEBUG_VARIABLE:
        print("orderBy: ", orderBy_clause)
    having_clause = having_to_SQL(query["having"])
    if DEBUG_VARIABLE:
        print("having_clause: ", having_clause)
    limit_clause = limit_to_SQL(query["limit"])
    if DEBUG_VARIABLE:
        print("limit_clause: ", limit_clause)
    intersect_clause = intersect_except_union_to_SQL(query["intersect"], "INTERSECT", brackets=True)#always assume brackets. Then in the decoding, Remove em.
    except_clause = intersect_except_union_to_SQL(query["except"], "EXCEPT", brackets=True)
    union_clause = intersect_except_union_to_SQL(query["union"], "UNION", brackets=True)

    return select + "\n" + from_clause + "\n" + where_clause + "\n" + groupBy_clause + "\n" + orderBy_clause + "\n" + having_clause + "\n" + limit_clause + "\n" + intersect_clause + "\n" + except_clause + "\n" + union_clause + "\n"

#convert simple SQL to SQL
def SimpleSQL_to_SQL(query):
    #replace brackets in Intersect Union Except
    for x in ["INTERSECT ", "UNION ", "EXCEPT "]:#careful extra space here
        pattern = x+"(SELECT"
        pos = query.find(pattern)
        if pos != -1:#we found INTERSECT Query
            #get end of the intersect query
            pos += len(x) 
            yes, end_idx = is_subquery_complete(query[pos:])
            if yes == False:#error
                return "SELECT ERROR"
            query = query.replace(query[pos:pos+end_idx+2],query[pos+1:pos+end_idx])#+2 one is bracket, 1 is \n 
    #remove all lines containing none
    query_split = query.split("\n")
    valid_lines = []
    for line in query_split:
        if "NONE" in line:
            pass
        else:
            valid_lines.append(line)
    
    return " ".join(valid_lines)



#DO THE REALTIME CHECKS
def get_t1c1_from_word(word):
    t1c1 = word.split(".")#always assume only two 
    if len(t1c1)!=2:
        return "LengthWasNot2", "LengthWasNot2"
    t1=t1c1[0].lower().strip()
    c1=t1c1[1].lower().strip()
    return [t1, c1]

def t1c1_in_schema(t1, c1, schema):
    table_names = schema.keys()
    if t1.lower() not in table_names:
        return False
    if c1.lower() not in schema[t1.lower()]:
        return False
    return True

def validate_select(schema, select_clause):
    global error_Type
    #split the select clause into tokens
    #return all table.column names used in the select clause
    t1c1_names = []
    select_clause = select_clause.replace(","," ").replace(". ",".").replace("("," ( ").replace(")"," ) ")#get rid of commas#and that annoying join bug
    
    tokens = select_clause.split(" ") #might 

    if tokens[0] != "SELECT" and len(tokens) > 1:
        if error_Type == 0:
            error_Type = 3 #0 - no error, 1 wrong types example, 2 runtime, 3 syntax
        return False, []
    #token must be in agg operator, select, or in column names
    for token in tokens[1:-1]:#all tokens except the first and the last
        if token == "*":
            continue
        if token.lower() in AGG_OPS or token.lower() in ["(",")","distinct",""]:#aggr, open/close bracket, DISTINCT. I think this is wrong. fix later
            continue
        #not select or agg operator
        #must be t1c1
        t1, c1 = get_t1c1_from_word(token)
        if t1c1_in_schema(t1,c1,schema):
            t1c1_names.append([t1, c1])
        else:
            error_Type = 2 #0 - no error, 1 wrong types example, 2 runtime, 3 syntax
            if t1 == "LengthWasNot2":
                error_Type = 3
            return False, []
    return True, t1c1_names

def replace_string(text):
    #returns text and whether quote mark is open
    quote_marks = ["'",'"']
    for quote_mark in quote_marks:
        start_idx = 0
        while text[start_idx:].count(quote_mark) > 1:
            idx1 = text.find(quote_mark, start_idx)
            idx2 = text.find(quote_mark, idx1+1)
            text = text[:idx1]+'"text"'+text[idx2+1:]
            start_idx = len(text[:idx1]+'"text"')
            #breakpoint()
            
    for quote_mark in quote_marks:
        if text.count(quote_mark) == 1:
            return text, True
    return text, False



all_keywords = []
all_keywords.extend(CLAUSE_KEYWORDS)
all_keywords.extend(JOIN_KEYWORDS)
all_keywords.extend(WHERE_OPS)
all_keywords.extend(UNIT_OPS)
all_keywords.extend(AGG_OPS)
all_keywords.extend(COND_OPS)
all_keywords.extend(SQL_OPS)
all_keywords.extend(ORDER_OPS)
all_keywords.append("BY")
all_keywords.append("VALID")
all_keywords.append('')
all_keywords.append("!")
all_keywords = [w.lower() for w in all_keywords]

def validate_from(schema, from_clause, select_columns, require_all = False, start_word = "FROM"):
    global error_Type
    #first replace all strings with ("text") if w r in the where clause
    if start_word == "WHERE":
        from_clause, open_quote_mark = replace_string(from_clause)
        if open_quote_mark:
            #print("open quote mark")
            #breakpoint()
            return True
    #there's annoying thing with space:

    from_clause = from_clause.replace(". ",".") #almost always we can do this.
    #since > and = ... are valid even if they're not surrounded by spaces, add spaces around them
    for comparison in COMPARISONS:
        from_clause = from_clause.replace(comparison, " "+comparison+" ")

    all_valid_words = []
    all_valid_words.extend(all_keywords)
    all_valid_words.extend(((schema).keys()))
    #Make sure first word is correct
    words = from_clause.split(" ")
    if len(words) == 1:
        return True # need more than 1 word
    if start_word != words[0]:
        if error_Type == 0:
            error_Type = 2 #0 - no error, 1 wrong types example, 2 runtime, 3 syntax
        return False
    
    for word in words[1:-1]:
        word = word.replace(",","").replace("(","").replace(")","")
        #first check if the word is a float if we r in where clause
        if start_word == "WHERE":
            #word is a float
            try:
                float(word) #if word is a float we good
                continue
            except:
                pass
            #check if word is a string
            if len(word) == 0 or word[0] in ["'",'"']:
                continue
        
        if word.find(".")!=-1: #means t1.c1 (except in where where it might be a flaot)
            t1, c1 = get_t1c1_from_word(word)
            #check that t1,c1 are in schema (or in where clause check that they are not digits. bcs 15.00 is allowed, hope no strings are sentences)
            if t1c1_in_schema(t1,c1,schema)==False:# and t1.isdigit() == False and c1.isdigit() == False
                if error_Type == 0:
                    error_Type = 2 #0 - no error, 1 wrong types example, 2 runtime, 3 syntax
                return False
        elif word.lower() not in all_valid_words: #check the dictionary :) in from clause also allow 
            #print("not in dictionary")
            #print("FROM CLAUSE")
            #print(from_clause)
            #print("WORD")
            #print(word)
            #breakpoint()
            error_Type = 3
            return False
    #require all - means we require all tables to appear in FROM CLAUSE.
    if require_all:
        tables = set([table[0].lower() for table in select_columns])#set of tables.
        words = set([word.lower() for word in words])
        if tables.issubset(words):
            return True
        else:
            #breakpoint()#just trust it
            if error_Type == 0:
                error_Type = 2 #0 - no error, 1 wrong types example, 2 runtime, 3 syntax
            return False
    return True

def is_subquery_complete(subquery):
    #checks if at any point the number of open brackets equals the number of close brackets.
    o_b_s = [""]
    c_b_s = [""]
    for idx, ch in enumerate(subquery):
        if ch=="(":
            o_b_s.append("(")
        elif ch==")":
            c_b_s.append(")")
        if len(o_b_s) == len(c_b_s):#we have a full subquery!
            return True, idx
    return False, 0

def check_example_types(select_clause, example, db_full, schema):
    global error_Type
    #"SELECT count( * )"
    #get rid of all irrelevant stuff
    selected_cols = []
    #first split column selection by commas.
    query_parts = select_clause.split(",")
    #if there's an aggregation operator, we know what type it is
    for query_part in query_parts:
        query_part = query_part.replace(". ",".")
        agg_op_in_query = False
        for agg in AGG_OPS_NUMBER:#check if the string contains any of the aggregation operators.
            if agg+"(" in query_part or agg+" " in query_part:
                selected_cols.append("number")
                agg_op_in_query = True
                break
        if "time" in db_full["column_types"] and agg_op_in_query:
            tokens = query_part.split(" ")
            for token in tokens:
                li = get_t1c1_from_word(token.replace(",",""))
                if li in db_full["column_names_original"]:
                    idx = db_full["column_names_original"].index(li)
                    column_type = db_full["column_types"][idx]
                    if column_type == "time" and agg!="count" and agg!="sum":
                        selected_cols[-1] = "time"
        if agg_op_in_query == False:#didnt find agg, so search for column name
            tokens = query_part.split(" ")
            for token in tokens:
                if token == "*":
                    selected_cols.append("text")
                    return True # Impossible to check without join clause >_<. Could replace with the correct join in the future but this would delay the query checking
                else:
                    li = get_t1c1_from_word(token.replace(",",""))
                    if li in db_full["column_names_original"]:
                        idx = db_full["column_names_original"].index(li)
                        column_type = db_full["column_types"][idx]
                        selected_cols.append(column_type)
    cols_in_example = []
    for item in example:
        if item == "null" or item == None:
            cols_in_example.append("null")# or none
        elif isinstance(item, str):
            cols_in_example.append("text")
        else:
            cols_in_example.append("number")
    #print("cols_in_example: ", (cols_in_example))
    #print("selected_cols: ", (selected_cols))
    for true_col, pred_col in zip(cols_in_example, selected_cols):
        if true_col == "null": #null can be number or text ._.
            continue
        if true_col == pred_col or pred_col == "time" or pred_col=="other":#pred_col uses schema info
            continue
        error_Type = 1 #0 - no error, 1 wrong types example, 2 runtime, 3 syntax
        return False
    return True
    

#validate simple SQL
def validate_partial_SQL(schema, string, example=(), db_full = {}, return_error_type = False, break_early = False):
    if string=="":
        return True
    global error_Type
    error_Type = 0
    
    ## SUBQUERIES ##
    #check if we can find another (SELECT
    sq_begin = string.find("(SELECT")
    if sq_begin != -1:#then check if there is a matching bracket (I.e. the highest level select is closed.)
        is_complete, end_idx = is_subquery_complete(string[sq_begin:])
        
        if is_complete:#we have a complete subquery that we assume is valid. Replace it with the word valid.
                #we verify if the subquery is valid.
                complete_subquery = string[sq_begin+1:sq_begin+end_idx]
                error_type = validate_partial_SQL(schema, complete_subquery, example=(), db_full = db_full, return_error_type = True)#pass in the subquery into validate_partial_SQL!    
                if error_type==0:
                    #let's replace complete_subquery with VALID
                    string = string.replace("("+complete_subquery," VALID ")#if subquery is right, replace with VALID
                else:
                    if break_early:
                        breakpoint()
                    if return_error_type:
                        return error_type
                    else:
                        return False
        else:#query not complete, recursion to the rescue!
            error_type = validate_partial_SQL(schema, string[sq_begin+1:], example=(), db_full = db_full, return_error_type = True)#pass in the subquery into validate_partial_SQL!
            if break_early:
                breakpoint()
            if return_error_type:
                return error_type
            else:
                if error_type == 0:
                    return True
                return False
            
                
    string_broken = string.split("\n")
    nr_clauses = len(string_broken)

    
    #check SELECT
    if nr_clauses > 1:
        select_valid, select_columns = validate_select(schema, string_broken[0]+" ")#we add a space, since by default we do not check the final token, as it may be partially generated.
    else:
        select_valid, select_columns = validate_select(schema, string_broken[0])
    #check FROM
    from_valid = True
    if nr_clauses > 1:
        if nr_clauses > 2:
            from_valid = validate_from(schema, string_broken[1]+" ", select_columns, require_all = True, start_word = "FROM") 
        else:
            from_valid = validate_from(schema, string_broken[1], select_columns, require_all = False, start_word = "FROM")#all replaced with validate from.
    #check where 
    where_valid = True
    if nr_clauses > 2:
        if nr_clauses > 3:
            where_valid = validate_from(schema, string_broken[2]+" ", select_columns, require_all = False, start_word = "WHERE")
        else:
            where_valid = validate_from(schema, string_broken[2], select_columns, require_all = False, start_word = "WHERE")
    #check group by
    groupBy_valid = True
    if nr_clauses > 3:
        if nr_clauses > 4:
            groupBy_valid = validate_from(schema, string_broken[3]+" ", select_columns, require_all = False, start_word = "GROUP")
        else:
            groupBy_valid = validate_from(schema, string_broken[3], select_columns, require_all = False, start_word = "GROUP")
    #check order by
    orderBy_valid = True
    if nr_clauses > 4:
        if nr_clauses > 5:
            orderBy_valid = validate_from(schema, string_broken[4]+" ", select_columns, require_all = False, start_word = "ORDER")
        else:
            orderBy_valid = validate_from(schema, string_broken[4], select_columns, require_all = False, start_word = "ORDER")
    
    #do example checking
    examples_valid = True
    if example != () and (select_valid and from_valid and where_valid and groupBy_valid and orderBy_valid) and nr_clauses > 1: #we need to get the right column types
        examples_valid = check_example_types(string_broken[0], example, db_full, schema)
    #if string.find("SELECT document_id, documents.document_name, documents.document_description")!=-1:
    #    breakpoint()

    #INTERSECT, EXCEPT, UNION CHECKED RECURSIVELY.
    if break_early:
        breakpoint()
    if return_error_type:
        if error_Type == 0 and ((select_valid and from_valid and where_valid and groupBy_valid and orderBy_valid and examples_valid)==False):
            breakpoint()
            print("No error, yet the query is invalid. WHY?")
        return error_Type
    else:
        #print("Got here")
        #print(select_valid , from_valid , where_valid , groupBy_valid , orderBy_valid , examples_valid)
        return select_valid and from_valid and where_valid and groupBy_valid and orderBy_valid and examples_valid
    #takes in partial SQL and returns if it is valid.


def get_1_off_queries(query, db_full={}):
    query = query.lower()
    #for every off by 1 error, return it.
    AGG_OPS = ['', 'max', 'min', 'count', 'sum', 'avg']#none is replaced
    all_new_queries = []
    query_split = query.split(" ")
    for idx, token in enumerate(query_split):
        #aggregation
        for agg_cont in AGG_OPS[1:]:#AGGREGATION
            agg_idx = token.find(agg_cont+"(")
            if agg_idx != -1:
                for agg_perm in AGG_OPS:#replace it with all possible aggregation operations
                    cop = copy.deepcopy(query_split)
                    cop[idx] = token.replace(agg_cont+"(", agg_perm+"(")
                    all_new_queries.append(" ".join(cop))
        
        for where_op in WHERE_OPS:#WHERE
            if token.lower() == where_op.lower():
                for where_op in WHERE_OPS:#replace it with all possible aggregation operations
                    cop = copy.deepcopy(query_split)
                    cop[idx] = where_op
                    all_new_queries.append(" ".join(cop))
        
        for order_op in ORDER_OPS:#ORDER
            if token.lower() == order_op.lower():
                for order_op in ORDER_OPS:#replace it with all possible aggregation operations
                    cop = copy.deepcopy(query_split)
                    cop[idx] = order_op
                    all_new_queries.append(" ".join(cop))

        
        t1, c1 = get_t1c1_from_word(token.replace(",","")) #li also needs to be in the from CLAUSE
        li = [t1, c1]
        if li in db_full["column_names_original"]:
            for li in db_full["column_names_original"][1:]:
                #table name must appear in query somewhere
                if query.find(li[0])!=-1:
                    cop = copy.deepcopy(query_split)
                    cop[idx] = ".".join(li)
                    all_new_queries.append(" ".join(cop))
    return all_new_queries