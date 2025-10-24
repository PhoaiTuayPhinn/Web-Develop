function calc_factorial(n)
{
    fact=1
    for(i=1;i<=n;i++)
    {
        fact*=i
    }
    return fact
}
function use_factorial(n,result)
{
    fact=calc_factorial(n)
    result.innerText=fact
}