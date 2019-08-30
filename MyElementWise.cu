/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cuda_fp16.h>
#include <cassert>
#include <vector>
#include <set>
#include <cmath>
//#include <math.h>
#include "MyElementWise.hpp"

using namespace std;

bool gdb_copy_to_cpu(std::string opType, float* data, int size)
{
  float* _temp = (float*)malloc(size*sizeof(float));
  
  cudaError_t status = (cudaMemcpy(_temp, data, size*sizeof(float), cudaMemcpyDeviceToHost));

  if( status != cudaSuccess ) { 
     printf("gdb_copy_to_cpu cuda error! \n");
     return false;
  } 

  printf("gdb_copy_to_cpu --- \n");

  free(_temp);
  return true;
}


//sds:对于这个split来说，这里的index必须是0
//sds: index,The index of the output tensor.
nvinfer1::Dims MyElementWisePlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  // 'not' is 1, 'equal' and 'less' is 2, 'where' is 3
  assert(nbInputs == 1 || nbInputs == 2 || nbInputs == 3);
  assert(index == 0);// only one output
  //output_dims == input_dims[0]
  
  if(nbInputs == 1)
  {
      output_dims = inputDims[0];
  }
  else
  {
      //检查rank要一致
      assert(inputDims[0].nbDims == inputDims[1].nbDims);
      if(nbInputs == 3)
          assert(inputDims[2].nbDims == inputDims[1].nbDims);
      vector<set<int>> _sets;
      _sets.resize(inputDims[0].nbDims);
      //broadcast,选择维度最多的作为输出
      int max_num =0;
      int max_index = 0;
      for(int i = 0; i < nbInputs; i++)
      {

          nvinfer1::Dims const& _temp = inputDims[i];
          int total_num =0;
          for(int j=0; j<_temp.nbDims;j++)
          {
             int d_num =  _temp.d[j];
             _sets[j].insert(d_num);
             total_num +=d_num;
          }

          if(max_num < total_num)
          {
              max_num = total_num;
              max_index = i;
          }
      }

      //判断是否满足broadcast条件
      for(int i=0; i< _sets.size(); i++)
      {
          set<int>& _set = _sets[i];
          if(_set.size() != 1 && _set.size() !=2)
            cout << "MyElementWisePlugin->getOutputDimensions(), the dims must allow broadcast 1! error!" << endl;
          //两个值，必定有一个是1
          if(_set.size() ==2 && !_set.count(1))
            cout << "MyElementWisePlugin->getOutputDimensions(), the dims must allow broadcast 2! error!" << endl;
            
      }      
            
      //sds-temp,返回set.max * set.max *set.max
      output_dims = inputDims[max_index];       
      for(int i = 0; i< _sets.size(); i++)
      {
         //sets是排序的，直接取第二个.
         output_dims.d[i] = *(++_sets[i].begin());
      }
          
  }
  return output_dims;
}

int MyElementWisePlugin::initialize() {
  //nvinfer1::Dims dims = this->getOutputDims(0);
  int nbInputs = this->getInputDimsNum();



    if(nbInputs == 1)
      {
          output_dims = this->getInputDims(0);
      }
      else
      {
          //检查rank要一致
          assert(this->getInputDims(0).nbDims == this->getInputDims(1).nbDims);
          if(nbInputs == 3)
              assert(this->getInputDims(2).nbDims == this->getInputDims(1).nbDims);
          vector<set<int>> _sets;
          _sets.resize(this->getInputDims(0).nbDims);
          //broadcast,选择维度最多的作为输出
          int max_num =0;
          int max_index = 0;
          for(int i = 0; i < nbInputs; i++)
          {
    
              nvinfer1::Dims const& _temp = this->getInputDims(i);
              int total_num =0;
              for(int j=0; j<_temp.nbDims;j++)
              {
                 int d_num =  _temp.d[j];
                 _sets[j].insert(d_num);
                 total_num +=d_num;
              }
    
              if(max_num < total_num)
              {
                  max_num = total_num;
                  max_index = i;
              }
          }
    
          //判断是否满足broadcast条件
          for(int i=0; i< _sets.size(); i++)
          {
              set<int>& _set = _sets[i];
              if(_set.size() != 1 && _set.size() !=2)
                cout << "MyElementWisePlugin->getOutputDimensions(), the dims must allow broadcast 1! error!" << endl;
              //两个值，必定有一个是1
              if(_set.size() ==2 && !_set.count(1))
                cout << "MyElementWisePlugin->getOutputDimensions(), the dims must allow broadcast 2! error!" << endl;
                
          }      
                
          //sds-temp,返回set.max * set.max *set.max
          output_dims = this->getInputDims(max_index);       
          for(int i = 0; i< _sets.size(); i++)
          {
             //sets是排序的，直接取第二个.
             output_dims.d[i] = *(++_sets[i].begin());
          }
              

     }



  
    //设置变量，kernel中会用到 
  rank =  this->getInputDims(0).nbDims;
  h_dims_a.resize(rank);
  h_dims_a_mul.resize(rank);
  h_dims_b.resize(rank);
  h_dims_b_mul.resize(rank);
  h_dims_c.resize(rank);
  h_dims_c_mul.resize(rank);
  h_dims_y.resize(rank);
  h_dims_y_mul.resize(rank);

  
  for(int i = 0 ; i<rank; i++)
  {
    h_dims_a[i] = this->getInputDims(0).d[i];
    if(nbInputs == 2 || nbInputs == 3)
       h_dims_b[i] = this->getInputDims(1).d[i];
    if(nbInputs == 3)
        h_dims_c[i] = this->getInputDims(2).d[i];

    h_dims_y[i] = output_dims.d[i];

  }
  h_dims_a_mul[rank-1]=1;
  h_dims_b_mul[rank-1]=1;
  h_dims_c_mul[rank-1]=1;
  h_dims_y_mul[rank-1]=1;
  for(int i = rank-2; i>=0 ; i--)
  {
      h_dims_a_mul[i]=h_dims_a_mul[i+1]*h_dims_a[i+1];
      h_dims_b_mul[i]=h_dims_b_mul[i+1]*h_dims_b[i+1];
      if(nbInputs == 3)
          h_dims_c_mul[i]=h_dims_c_mul[i+1]*h_dims_c[i+1];
      h_dims_y_mul[i]=h_dims_y_mul[i+1]*h_dims_y[i+1];
  }

      
  _numbers = 1;
  for( int i=output_dims.nbDims-1; i>=0; i-- ) {
    _numbers *= output_dims.d[i];
  }

  //赋值给device vetor

    dims_a     =  h_dims_a        ;
    dims_a_mul =  h_dims_a_mul   ;
    dims_b     =  h_dims_b       ;
    dims_b_mul =  h_dims_b_mul   ;
    dims_c     =  h_dims_c       ;
    dims_c_mul =  h_dims_c_mul   ;
    dims_y     =  h_dims_y       ;
    dims_y_mul =  h_dims_y_mul   ;
  
  return 0;
}


/*
Returns the tensor resulted from performing the equal logical operation elementwise on the input tensors A and B (with Numpy-style broadcasting support).

This operator supports multidirectional (i.e., Numpy-style) broadcasting;

*/

//sds-temp，支持多维度扩展
template<typename T>
__global__ void equal_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y, int rank, int* aDim, int* aDimMul, int* bDim, int* bDimMul, int* yDim, int* yDimMul) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        int a_index = 0;
        int b_index = 0;
        int tmp_index = index;
        for(int j = 0 ; j < rank; j++)
        {
            int temp1 = tmp_index / yDimMul[j];
            tmp_index = tmp_index % yDimMul[j];

            {
                a_index += (temp1 % aDim[j])* aDimMul[j];
                b_index += (temp1 % bDim[j])* bDimMul[j];
            }

        }

        a_index += tmp_index;
        b_index += tmp_index;
        
            
        T temp=0;
        if(fabs(a[a_index] - b[b_index]) < 1e-6)
            temp = 1;
        y[index] = temp;
    }
  }



template<typename T>
__global__ void less_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y, int rank, int* aDim, int* aDimMul, int* bDim, int* bDimMul, int* yDim, int* yDimMul) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        int a_index = 0;
        int b_index = 0;
        int tmp_index = index;
        for(int j = 0 ; j < rank; j++)
        {
            int temp1 = tmp_index / yDimMul[j];
            tmp_index = tmp_index % yDimMul[j];

            {
                a_index += (temp1 % aDim[j])* aDimMul[j];
                b_index += (temp1 % bDim[j])* bDimMul[j];
            }
      
        }

        a_index += tmp_index;
        b_index += tmp_index;
        
            
        T temp=0;
        if((int)a[a_index] < (int)b[b_index])
            temp = 1;
        y[index] = temp;
    }
  }

template<typename T>
__global__ void where_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T const* __restrict__ c, T * __restrict__ y, 
    int rank, int* aDim, int* aDimMul, int* bDim, int* bDimMul, int* cDim, int* cDimMul, int* yDim, int* yDimMul) {
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        int a_index = 0;
        int b_index = 0;
        int c_index = 0;
        int tmp_index = index;
        for(int j = 0 ; j < rank; j++)
        {
            int temp1 = tmp_index / yDimMul[j];
            tmp_index = tmp_index % yDimMul[j];
            //if(j!= rank-1)
            {
                a_index += (temp1 % aDim[j])* aDimMul[j];
                b_index += (temp1 % bDim[j])* bDimMul[j];
                c_index += (temp1 % cDim[j])* cDimMul[j];
            }
            //else
            //{
             //   a_index += (tmp_index % aDim[j]);
             //   b_index += (tmp_index % bDim[j]);
             //   c_index += (tmp_index % cDim[j]);

           // }
        }

            
        if((int)a[a_index])
        {
            y[index] =b[b_index];
        }
        else
        {
            y[index] =c[c_index];
        }
    }
  }

template<typename T>
//equal 0 ,return 1 . else ,return 0
__global__ void not_kernel(const int n, T const* __restrict__ a, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        T temp=0;
        if(fabs(a[index] - 0.0) <= 1e-6)
            temp = 1;
        y[index] = temp;
    }
  }

//sds-temp, add mul pow 未使用，以下作废
template<typename T>
__global__ void add_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        //sds-temp,未考虑溢出
        y[index] = a[index] + b[index];
    }
  }

template<typename T>
__global__ void mul_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        //sds-temp,未考虑溢出
        y[index] = a[index] * b[index];
    }
  }

template<typename T>
__global__ void pow_kernel(const int n, T const* __restrict__ a,
    T const* __restrict__ b, T * __restrict__ y) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x)
    {
        //sds-temp,未考虑溢出
        y[index] = a[index] + b[index];
    }
  }




//sds:这里的inputs是在显存，outputs是在内存。
//sds:每个plugin进入enqueue，带过来的inputs有可能在内存或者显存，由addPluginV2时传入的inputs决定
//    inputs是一个指针，维度信息需要自己初始化，比如在initialize中或者enqueue中。
//    outputs? outputs已经按照getOutputDims指定的初始化，应该都是gpu指针吧?
int MyElementWisePlugin::enqueue(int batchSize,
                         const void *const *inputs, void **outputs,
                         void *workspace, cudaStream_t stream) {

  float  const* idata1    = reinterpret_cast<float  const*>(inputs[0]);
  

  float * odatas = reinterpret_cast<float *>(outputs[0]);

  /*
  cudaError_t cuda_status =
    cudaMemcpyAsync(odatas, h_odatas,
                    _d_output_ptrs.size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);
  if( cuda_status != cudaSuccess ) {
    return 1;
  }
  */

  dim3 block(512);
  dim3 grid((_numbers + 512 - 1) / 512);

  int * p_dims_a =thrust::raw_pointer_cast(&dims_a[0]);
  int * p_dims_a_mul =thrust::raw_pointer_cast(&dims_a_mul[0]);
  int * p_dims_b =thrust::raw_pointer_cast(&dims_b[0]);
  int * p_dims_b_mul =thrust::raw_pointer_cast(&dims_b_mul[0]);
  int * p_dims_c =thrust::raw_pointer_cast(&dims_c[0]);
  int * p_dims_c_mul =thrust::raw_pointer_cast(&dims_c_mul[0]);
  int * p_dims_y =thrust::raw_pointer_cast(&dims_y[0]);
  int * p_dims_y_mul =thrust::raw_pointer_cast(&dims_y_mul[0]);
  
 switch(_type){
     
  case(MyElementWiseType::Equal): 
  {
    float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);

    gdb_copy_to_cpu("Equal input1", (float *)idata1, h_dims_a[0]*h_dims_a_mul[0]);
    gdb_copy_to_cpu("Equal input2", (float *)idata2, h_dims_b[0]*h_dims_b_mul[0]);

    
    equal_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2, odatas, rank, p_dims_a, p_dims_a_mul, p_dims_b, p_dims_b_mul, p_dims_y, p_dims_y_mul);


    

    gdb_copy_to_cpu("Equal output", (float *)odatas, h_dims_y[0]*h_dims_y_mul[0]);
    break;
  }
  case(MyElementWiseType::Less):
    
  {
    float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);


    
     gdb_copy_to_cpu("Less input1", (float *)idata1, h_dims_a[0]*h_dims_a_mul[0]);
    gdb_copy_to_cpu("Less input2", (float *)idata2, h_dims_b[0]*h_dims_b_mul[0]);

    
    less_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2, odatas, rank, p_dims_a, p_dims_a_mul, p_dims_b, p_dims_b_mul, p_dims_y, p_dims_y_mul);





    gdb_copy_to_cpu("Less output", (float *)odatas, h_dims_y[0]*h_dims_y_mul[0]);
    break;
  }
  case(MyElementWiseType::Where):
  {
    float  const* idata2    = reinterpret_cast<float  const*>(inputs[1]);
    float  const* idata3    = reinterpret_cast<float  const*>(inputs[2]);

    
       gdb_copy_to_cpu("Where input1", (float *)idata1, h_dims_a[0]*h_dims_a_mul[0]);
    gdb_copy_to_cpu("Where input2", (float *)idata2, h_dims_b[0]*h_dims_b_mul[0]);
    gdb_copy_to_cpu("Where input3", (float *)idata3, h_dims_c[0]*h_dims_c_mul[0]);

    
    where_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, idata2,idata3, odatas, rank, p_dims_a, p_dims_a_mul,
        p_dims_b, p_dims_b_mul,p_dims_c, p_dims_c_mul, p_dims_y, p_dims_y_mul);




    gdb_copy_to_cpu("Where output", (float *)odatas, h_dims_y[0]*h_dims_y_mul[0]);

    
    break;
  }w
  case(MyElementWiseType::Not):
  {

    gdb_copy_to_cpu("Not input1", (float *)idata1, h_dims_a[0]*h_dims_a_mul[0]);
    not_kernel<<<grid, block, 0, stream>>>(_numbers, idata1, odatas);
 

    gdb_copy_to_cpu("Not output", (float *)odatas, h_dims_y[0]*h_dims_y_mul[0]);
    break;
  }
  default:
    break;
  }
  return cudaGetLastError() != cudaSuccess;
}
