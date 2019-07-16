/**
 * Transpose test.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdio.h>

#include "transpose.h"

int main(void)
{
    int rc = test_transpose();
    printf("%s\n", rc ? "Failed" : "Success");
    return rc;
}
