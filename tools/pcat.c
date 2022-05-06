#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "libpq-fe.h"

static void
exit_nicely(PGconn *conn)
{
    PQfinish(conn);
    exit(1);
}

int
main(int argc, char **argv)
{
    const char *conninfo;
    PGconn     *conn;
    PGresult   *res;
    int         nFields, nTuples;
    int         i,
                j,
                r;

    if (argc <= 1) {
        fprintf(stderr, "usage: %s <SQL>\n", argv[0]);
        exit(1);
    }

    char const *SQL = argv[1];

    /* Make a connection to the database */
    conn = PQconnectdb("");

    /* Check to see that the backend connection was successfully made */
    if (PQstatus(conn) != CONNECTION_OK)
    {
        fprintf(stderr, "Connection to database failed: %s",
                PQerrorMessage(conn));
        exit_nicely(conn);
    }

    /*
     * Our test case here involves using a cursor, for which we must be
     * inside a transaction block.  We could do the whole thing with a
     * single PQexec() of "select * from pg_database", but that's too
     * trivial to make a good example.
     */

    /* Start a transaction block */
    r = PQsendQuery(conn, SQL);
    if (r != 1) {
        fprintf(stderr, "SQL failed: %s", PQerrorMessage(conn));
        PQclear(res);
        exit_nicely(conn);
    }

    r = PQsetSingleRowMode(conn);
    if (r != 1) {
        fprintf(stderr, "failed to enter single row mode: %s", PQerrorMessage(conn));
        exit_nicely(conn);
    }

    for (;;) {

        res = PQgetResult(conn);
        if (res == NULL) break;
        nTuples = PQntuples(res);
        nFields = PQnfields(res);
        for (i = 0; i < nTuples; i++)
        {
            for (j = 0; j < nFields; j++) {
                if (j) {
                    putc(',', stdout);
                }
                fputs(PQgetvalue(res, i, j), stdout);
            }
            putc('\n', stdout);
        }

        PQclear(res);
    }

    PQfinish(conn);

    return 0;
}
