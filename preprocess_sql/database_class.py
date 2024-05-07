from preprocess_sql.parse_raw_json import get_schemas_from_json

class DatabaseClass():
    def __init__(self, db_path: str):
        schemas, db_names, tables, dbs_full_schema = get_schemas_from_json(db_path)
        self.schemas = schemas
        self.db_names = db_names
        self.tables = tables
        self.dbs_full_schema = dbs_full_schema
        self.process_schema()

    def process_schema(self):
        for key in self.dbs_full_schema.keys():
            db_full = self.dbs_full_schema[key]
            for idx, column_name in enumerate(db_full['column_names_original']):
                #the first element of column name is the table
                table_idx = column_name[0]
                if table_idx == -1:#skip first token
                    continue
                table_name = db_full["table_names_original"][table_idx]
                db_full['column_names_original'][idx][0] = table_name.lower()
                db_full['column_names_original'][idx][1] = db_full['column_names_original'][idx][1].lower()