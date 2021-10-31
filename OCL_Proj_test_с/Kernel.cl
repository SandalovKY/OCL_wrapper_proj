__kernel void operation(__global int * inout)
{
    inout += get_global_id(0);
}