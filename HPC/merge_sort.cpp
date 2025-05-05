#include<iostream>
#include<omp.h>
using namespace std;

void displayArray(int arr[], int size)
{
    cout<<"Array : [ ";
    for(int i = 0; i < size; i++)
    {
        cout<<arr[i]<<" ";
    }
    cout<<" ]"<<endl;
}

void merge(int nums[], int leftstart, int leftend, int rightstart, int rightend )
{
    int n = (rightend - leftstart) + 1;

    int temp[n];

    int t = 0;

    int l = leftstart;
    int r = rightstart;

    while(l <= leftend && r <= rightend)
    {
        if(nums[l] <= nums[r])
        {
            temp[t++] = nums[l++];
        }
        else
        {
            temp[t++] = nums[r++];
        }
    }

    while(l <= leftend)
    {
        temp[t++] = nums[l++];
    }

    while(r <= rightend)
    {
        temp[t++] = nums[r++];
    }
    
    for(int i = 0; i < n; i++)
    {
        nums[leftstart + i] = temp[i];
    }
}

void mergeSort(int nums[], int start, int end)
{
    if(start < end)
    {
        int mid = (start + end)/2;
        
        #pragma omp parallel sections
        {
            #pragma omp section
                mergeSort(nums , start , mid);
            #pragma omp section
                mergeSort(nums, mid+1, end);
        }
        merge(nums, start, mid, mid+1, end);
    }
}

void bubbleSort(int nums[], int length)
{
    for(int i = 0; i < length; i++)
    {
        int start = i % 2;
        #pragma omp parallel for
        for(int j  = start; j < length - 1; j = j + 2)
        {
            if(nums[j] >= nums[j + 1])
            {
                swap(nums[j], nums[j+1]);
            }
        }
    }
}

int main()
{
    int nums[] = {9,1,2,6,4,5,8,5,4,6,1,2,5};

    int length = sizeof(nums) / sizeof(int);

    cout<<"Parallel Merge Sort : "<<endl;
    cout<<"Before Sorting : ";
    displayArray(nums , length);

    mergeSort(nums , 0 , length - 1);

    cout<<"After Sorting : ";
    displayArray(nums , length);

    int nums1[] = {9,1,2,6,4,5,8,5,4,6,1,2,5};

    int length1 = sizeof(nums1) / sizeof(int);
    cout<<"Parallel Bubble Sort : "<<endl;
    cout<<"Before Sorting : ";
    displayArray(nums1 , length1);
    bubbleSort(nums1 , length1);
    cout<<"After Sorting : ";
    displayArray(nums1 , length1);

    return 0;
}