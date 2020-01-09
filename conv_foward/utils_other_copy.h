#ifndef UTILS_OTHER
#define UTILS_OTHER

/** 
 * \brief  为卷积核及对应输出、激活层和池化等分配空间
 *  
 * \param[in] weight           卷积核参数
 * \param[in] ksize_m          x方向卷积核尺寸
 * \param[in] ksize_n          y方向卷积核尺寸
 * \param[in] input_c          输入数据通道数
 * \param[in] out_c            输出数据通道数
 * 
 */
template<class T>
T ****get_mem(float* weight,int ksize_m, int ksize_n, int input_c, int out_c)
{
    T ****p4_a = new T ***[ksize_m];
    for(int i=0; i<ksize_m; i++) {
        p4_a[i] = new T **[ksize_n];
        for(int j=0; j<ksize_n; j++) {
            p4_a[i][j] = new T *[input_c];
            for(int k=0; k<input_c; k++) {
                p4_a[i][j][k] = new T [out_c];
                for(int l=0; l<out_c; l++) {
                    p4_a[i][j][k][l] = weight[i*ksize_n*input_c*out_c + j*input_c*out_c + k*out_c + l];
                }
            }
        }
    }
    return p4_a;
}

/** 
 * \brief  为卷积输出、激活输出和池化输出等分配空间
 *  
 * \param[in] batch 输入数据批次
 * \param[in] m     输入数据x方向尺寸
 * \param[in] n     输入数据y方向尺寸
 * \param[in] c     输出数据通道
 * 
 */
template<class T>
T ****get_mem(int batch, int m, int n, int c)
{
    T ****a = new T ***[batch];
    for(int i=0;i<batch;i++)
    {
        a[i] = new T **[m];
        for(int j=0;j<m;j++)
        {
            a[i][j] = new T *[n];
            for(int k=0;k<n;k++)
            {
                a[i][j][k] = new T [c];
                for(int p=0;p<c;p++)
                {
                    a[i][j][k][p] = 0;
                }
            }
        }
    }
    return a;
}

/** 
 * \brief  为卷积核及对应激活层和池化输出等释放空间
 *  
 * \param[in] p4_a  需要释放空间的四重指针
 * \param[in] batch 输入数据批次
 * \param[in] m     输入数据x方向尺寸
 * \param[in] n     输入数据y方向尺寸
 * \param[in] c     输出数据通道数
 */
template<class T>
void delete_mem(T ****p4_a, int batch, int m, int n, int c)
{
    for(int i=0; i<batch; i++) {
        for(int j=0; j<m; j++) {
            for(int k=0; k<n; k++) {
                delete [] p4_a[i][j][k];
            }
            delete p4_a[i][j];
        }
        delete p4_a[i];
    }
    delete p4_a;
}

#endif