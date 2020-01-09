#include <iostream>
#include "layer.h"
#include "utils_other_copy.h"

using namespace std;
static float img_test[5*5] = {1.0, 2.0, 3.0, 1.0, 2.0, 
                              3.0, 1.0, 2.0, 3.0, 1.0, 
                              2.0, 3.0, 1.0, 2.0, 3.0, 
                              1.0, 2.0, 3.0, 1.0, 2.0, 
                              3.0, 1.0, 2.0, 3.0, 4.0};
static float w1[3*3*1*2] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
static float b1[2] = {0.5,0.5};
static float ****p4_w1;
static float ****p4_img;

int main()
{
    CNN_layer conv1;
    p4_w1  = get_mem<float>(w1, 3, 3, 1, 2);
    p4_img = get_mem<float>(img_test, 1, 5, 5, 1);
    conv1.init(1,5,5,1,2,p4_w1,b1,3,3,1,1,VALID);
    conv1.forward(p4_img);
    printf("SAME:");
    for (int i=0; i<conv1.output_valid_y;i++) {
        printf("\n");
        for (int j=0; j<conv1.output_valid_x;j++) {
            printf("\n");
            for (int k=0; k<conv1.output_channel;k++) {
                /* 因为此处批次为1，故第一个索引为0 */
                printf("%lf ", conv1.out[0][i][j][k]);
            }
        }
    }
    conv1.init(1,5,5,1,2,p4_w1,b1,3,3,1,1,SAME);
    conv1.forward(p4_img);
    printf("\nvalid:");
    for (int i=0; i<conv1.input_y;i++) {
        printf("\n");
        for (int j=0; j<conv1.input_x;j++) {
            printf("\n");
            for (int k=0; k<conv1.output_channel;k++) {
                printf("%lf ", conv1.out[0][i][j][k]);
            }
        }
    }
    return 0;
}
