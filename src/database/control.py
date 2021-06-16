from .session import engine


def truncate(table):
        query = "TRUNCATE TABLE {table}"
        engine.execute(query.format(table=table))


def insert(table, data):
    # トランザクションログフルによるエラー回避のため、特定の件数ずつINSERT
    single_insert_volume = 100000
    insert_list = [data[i: i + single_insert_volume] for i in range(0, len(data), single_insert_volume)]
    for ins in insert_list:
        with engine.begin() as con:
            ins.to_sql(table, con=con, if_exists='append', index=False)
