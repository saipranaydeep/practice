#include <iostream>
using namespace std;

double power(double BaseNum, int PowNum){
    double result=1;
    for(int i=0; i<PowNum;i++){
        result = (double)result*BaseNum;
    }
    return result;
}

long factorial(int x){
	long res = 1;
	if(x == 0)
		return 1;
	for(int i=1; i<=x; i++) 
		res *= i;
	return res;
}

double sin(double x){
	int sign = 1;
	double total_sum = 0;
	for(int i=1; i<31; i += 2){
		total_sum += sign*(double)power(x,i)/factorial(i);
		sign = -1*sign;
	}
	return total_sum;
}


double cos(double x){
	int sign=1;
	double total_sum=0;
	for(int i=0; i<31; i+=2){
		total_sum += sign*(double)power(x,i)/factorial(i);
		sign= -1*sign;
	}
	return total_sum;
}

int main(){
   double num1,num2,rad,op,deg;
   double result;
   cout << "Hello! User" << endl;
   cout << "Welcome to P#A Calculator"<< endl;
   cout << "Here you can do simple calculations."<< endl;
    cout<<"______________________________________"<<endl;
    cout<<"|           *INSTRUCTIONS*           |"<<endl;
    cout<<"|'1'for Addition                     |"<<endl;
    cout<<"|'2'for Subraction                   |"<<endl;
    cout<<"|'3'for Multiplication               |"<<endl;
    cout<<"|'4'for Division                     |"<<endl;
    cout<<"|'5'for Power                        |"<<endl;
    cout<<"|'6'for Sine                         |"<<endl;
    cout<<"|'7'for Cosine                       |"<<endl;
    cout<<"|'8'for tangent                      |"<<endl;
    cout<<"|____________________________________|"<<endl;
    cout << "Enter your operator:"<< endl;
    cin >> op;
    if(op == 1){
        cout<<"Enter Your First Number"<<endl;
        cin>>num1;
        cout<<"Enter Your Second Number"<<endl;
        cin>>num2;
         result = num1 + num2;
      } else if(op == 2){
        cout<<"Enter Your First Number"<<endl;
        cin>>num1;
        cout<<"Enter Your Second Number"<<endl;
        cin>>num2;
         result = num1 - num2;
      }else if (op ==3){
        cout<<"Enter Your First Number"<<endl;
        cin>>num1;
        cout<<"Enter Your Second Number"<<endl;
        cin>>num2;
         result =num1 * num2;
      }else if(op == 4){
        cout<<"Enter Your Dividend"<<endl;
        cin>>num1;
        cout<<"Enter Your Divisor"<<endl;
        cin>>num2;
         result = (double)num1 / num2;
      }else if(op== 5){
        cout<<"Enter Your Base Number"<<endl;
        cin>>num1;
        cout<<"Enter Your Power"<<endl;
        cin>>num2;
         result =power(num1,num2);
      }else if(op== 6){
        cout<<"Enter the Angle in Degrees"<<endl;
        cin>>deg;
        rad=(deg*3.141592653589793238/180);
        result=sin(rad);
      }else if(op== 7){
        cout<<"Enter the Angle in Degrees"<<endl;
        cin>>deg;
        rad=(deg*3.141592653589793238/180);
        result=cos(rad);
      }else if(op== 8){
        cout<<"Enter the Angle in Degrees"<<endl;
        cin>>deg;
        rad=(deg*3.141592653589793238/180);
        result= sin(rad)/cos(rad);
      }else{
        cout<<"Pleaese Enter Valid Number"<<endl;
      }


      cout<<"Your ANSWER is"<<endl;
      cout << result;


    
//hello
    return 0;
}