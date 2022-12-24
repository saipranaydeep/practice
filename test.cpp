#include <iostream>
#include <fstream>
using namespace std;

int main()
{
    string a="Sai Pranay Deep";
    ofstream out ("Sai.txt");
    out<<a;

    return 0;

}