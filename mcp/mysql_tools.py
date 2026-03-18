from typing import Optional, Dict, Any, Annotated, List

import pymysql
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import json

mcp = FastMCP()

class Response(BaseModel):
    success: bool
    database: str
    table: str
    data: Optional[dict] | Optional[list]
    rowcount: Optional[int] = None
    error: Optional[str] = None

MYSQL_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'charset': 'utf8mb4',
}


def get_connection(db):
    config = MYSQL_CONFIG.copy()
    if db:
        config['database'] = db

    try:
        connection = pymysql.connect(**config)
        return connection
    except Exception as e:
        # msg = f'mysql connection error: {str(e)}'
        # return msg
        raise Exception(f'mysql connection error: {str(e)}')


def execute_query(command, database = None, params = None, commit = False):
    try:
        connection = get_connection(database)
        if not isinstance(connection, pymysql.Connection):
            return connection
        else:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(command, params)

                result = cursor.fetchall()

                if commit:
                    connection.commit()

                return result, cursor.rowcount
    except Exception as e:
        raise e


    #     response = Response(
    #         success=True,
    #         database=database,
    #         table='',
    #         data=tables,
    #         rowcount=rowcount,
    #     )
    #     return json.dumps(response.model_dump(), ensure_ascii=False)
    # except Exception as e:
    #     msg = f'list tables error: {str(e)}'
    #     return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_list_databases", description="列举MySQL中包含哪些数据库")
def mysql_list_databases():
    try:
        result, rowcount = execute_query("show databases")
        databases = [row['Database'] for row in result]
        response = Response(
            success=True,
            database="",
            table="",
            data=databases,
            rowcount=rowcount,
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'list tables error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_list_tables", description="获取指定数据库中的所有表")
def mysql_list_tables(database: str):
    try:
        result, rowcount = execute_query("show tables", database=database)
        tables = [list(row.values())[0] for row in result]
        response = Response(
            success=True,
            database=database,
            table='',
            data=tables,
            rowcount=rowcount,
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'list tables error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_describe_tables", description="获取表结构信息")
def mysql_describe_tables(database: str, table: str):
    try:
        result, rowcount = execute_query(f"describe {table}", database=database)
        response = Response(
            success=True,
            database=database,
            table=table,
            data=result,
            rowcount=rowcount,
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'describe tables error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_execute_query", description="执行SQL查询语句")
def mysql_execute_query(command, database=None, params: Optional[list] = None):
    try:
        params_tuple = tuple(params) if params else None
        result, rowcount = execute_query(command, database=database, params=params_tuple)
        response = Response(
            success=True,
            database=database,
            table='',
            data=result,
            rowcount=rowcount,
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'query tables error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)

@mcp.tool(name="mysql_insert_data", description="向表里插入数据")
def mysql_insert_data(database: str, table: str, data: Dict[str, str]):
    columns = list(data.keys())
    values = list(data.values())
    values_wrapper = ', '.join(['%s'] * len(values))
    command = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({values_wrapper})"

    try:
        result, rowcount = execute_query(command, database=database, params=tuple(values), commit=True)
        response = Response(
            success=True,
            database=database,
            table=table,
            data=result,
            rowcount=rowcount
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'insert tables error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_update_data", description="向表里更新数据")
def mysql_update_data(database: str, table: str, data: Dict[str, str], where: Dict[str, str]):
    set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
    where_clause = " and ".join([f"{k} = %s" for k in where.keys()])
    command = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

    set_params = list(data.values())
    where_params = list(where.values())
    params = set_params + where_params
    try:
        result, rowcount = execute_query(command, database=database, params=tuple(params), commit=True)
        response = Response(
            success=True,
            database=database,
            table=table,
            data=result,
            rowcount=rowcount
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'update tables error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_delete_data", description="向表里删除数据")
def mysql_delete_data(database: str, table: str, where: Dict[str, str]):
    where_clause = " and ".join([f"{k} = %s" for k in where.keys()])
    command = f"DELETE FROM {table} WHERE {where_clause}"

    params = list(where.values())
    try:
        result, rowcount = execute_query(command, database=database, params=tuple(params), commit=True)
        response = Response(
            success=True,
            database=database,
            table=table,
            data=result,
            rowcount=rowcount
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'delete tables error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_create_database", description="创建新数据库")
def mysql_create_database(database_name: str, charset: str = "utf8mb4"):
    command = f"CREATE DATABASE {database_name} CHARACTER SET {charset}"

    try:
        result, rowcount = execute_query(command)
        response = Response(
            success=True,
            database=database_name,
            table="",
            data=result,
            rowcount=rowcount,
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'create database error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_create_table", description="创建新表")
def mysql_create_table(
        database: str,
        table_name: str,
        table_columns: Annotated[str, Field(
            description="建表语句中的字段部分", 
            examples=[
                "`id` int NOT NULL AUTO_INCREMENT,`name` varchar(255) COLLATE utf8mb4_general_ci NOT NULL,PRIMARY KEY (`id`)"
            ]
        )],
        table_schema: Annotated[str, Field(
            description="建表语句中的补充部分", 
            examples=[
                "ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci"
			]
		)]                
):
    """
    建表语句示例：
    CREATE TABLE `user` (
        `id` int NOT NULL AUTO_INCREMENT,
        `name` varchar(255) COLLATE utf8mb4_general_ci NOT NULL,
        PRIMARY KEY (`id`)
    ) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
    :param database:
    :param table_name:
    :param table_columns:
    :param table_schema:
    :return:
    """
    command = f"CREATE TABLE `{table_name}` ({table_columns}) {table_schema}"

    try:
        result, rowcount = execute_query(command, database=database)
        response = Response(
            success=True,
            database=database,
            table=table_name,
            data=result,
            rowcount=rowcount,
        )
        return json.dumps(response.model_dump(), ensure_ascii=False)
    except Exception as e:
        msg = f'create table error: {str(e)}'
        return json.dumps({"success": False, "error": msg}, ensure_ascii=False)


@mcp.tool(name="mysql_execute_command", description="执行特定的SQL语句，如：变更表结构、增减字段")
def mysql_execute_command(database: str, command: str):
    try:
        result, rowcount = execute_query(command, database=database, commit=True)
        return Response(
            success=True,
            database=database,
            table="",
            data=result,
            rowcount=rowcount,
        )
    except Exception as e:
        msg = f'list databases error: {str(e)}'
        return msg


if __name__ == '__main__':
    mcp.run(transport="stdio")
    # print(mysql_list_databases())
    # print(mysql_list_tables('test'))
    # print(mysql_describe_tables('test', 'user'))
    # print(mysql_execute_query('select * from user where name=%s', database='test', params=["sam"]))
    # print(mysql_insert_data("test", "user", { "id": "5", "name": "someone3" }))
    # print(mysql_update_data("test", "user", { "name": "lily" }, { "id": "6", "name": "lucy" }))
    # print(mysql_delete_data("test", "user", { "id": "6" }))
    # print(mysql_create_table("test", "menu", "`id` int NOT NULL AUTO_INCREMENT, `name` varchar(255) NOT NULL, `price` decimal(10, 2), PRIMARY KEY (`id`)", "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci"))
