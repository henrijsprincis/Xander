prompt_sql = """
Your SQL query should follow the following rules:
1.) Each SQL statement should be on a separate line.

2.) You cannot use aliases in the SQL query. I.e. all column names should be written as table_name.column_name .

3.) You must use all of the following SQL keywords in your query: (SELECT, FROM, WHERE, GROUP BY, ORDER BY, HAVING, LIMIT, INTERSECT, UNION, EXCEPT).
When they are not needed, you should add NONE at the end of the line.

4.) You must seperate each word in the SQL query with a single space.

For example, the following queries are valid:

SELECT avg ( singer.age ), min ( singer.age )
FROM singer
WHERE singer.age > 20
GROUP BY NONE
ORDER BY NONE
HAVING NONE
LIMIT NONE
INTERSECT NONE
UNION NONE
EXCEPT NONE

SELECT department.name
FROM department
WHERE department.name = 'HR'
GROUP BY NONE
ORDER BY NONE
HAVING NONE
LIMIT NONE
INTERSECT ( SELECT department.name
FROM department
WHERE department.name = 'IT')

"""