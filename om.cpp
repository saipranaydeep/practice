#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
using namespace std;

class bank
{
    public:
    string name,password="sai";
    long long age=20,aadharNo,phoneNo,accountNo=rand() ,balace=0;
    void createAccount()
    {
        cout<<"Enter the details asked\n";
        cout<<"___________________________\n";

        cout<<"Enter Your Name: "; 
        getline(cin,name);
        cout<<"Enter your Age: ";
        cin>>age;
        if(age<18)
        {
            cout<<"You are not allowed to create an account\n";
        }
        else
        { 
            cout<<"Enter Your Phone Number: ";
            cin>>phoneNo;
            cout<<"Create a password: ";
            cin>>password;
            cout<<"Enter your Aadhar Number: ";
            cin>>aadharNo;
            cout<<"\n"<<"<<Your account has created successfully>>\n\n";
        }
    }
    void deposit()
    {
        long long dep;
        cout<<"Enter the amount of deposit: ";
        cin>>dep;
        cout<<"\n<<Amount "<<dep<<" has been deposited to your bank account successfully>>\n\n";
        balace+=dep;
    }
    void withdraw()
    {
        long long wd;
        cout<<"Enter the amount to withdraw: ";
        cin>>wd;
        if(balace>wd)
        {
            balace-=wd;
            cout<<"<<\nAmount "<<wd<<" has been withdrawn from your bank account successfully>>\n\n";   
        }
        else
        {
            cout<<"\n**Insufficient Balance**\n\n";    
        }
    }
    void details()
    {
        cout<<"\n\nName: "<<name<<"\n";
        cout<<"Age: "<<age<<"\n";
        cout<<"Contact: "<<phoneNo<<"\n";
        cout<<"Aadhar Number: "<<aadharNo<<"\n";
        cout<<"Account Number: "<<accountNo<<"\n\n\n";
    }
};

int main()
{
    bank user;
    int opt,Opt; 
    string entered_password;
    cout<<"_____________________________\n\n";
    cout<<"**WELCOME TO PRANAY BANK**\n";
    cout<<"_____________________________\n\n\n";
    cout<<"1: Create a new account\n";
    cout<<"2: Login to your account\n";
    cin>>Opt;
    if(Opt==1)
    {
        user.createAccount();
    }
    else if(Opt==2)
    {
        do
        { 
                cout<<"Enter your Password\n";
                cin>>entered_password;
                if(entered_password==user.password)
                {
                    do
                    {
                        if(user.age>18)
                        { 
                            cout<<"1: Check the Balance\n"<<"2: Deposit the money\n"<<"3: Withdrawl money\n"<<
                            "4: Show Details\n"<<"5: Exit\n\n"<<"Enter your option: ";
                            cin>>opt;

                            switch(opt)
                            {  
                                case 1:
                                {
                                    cout<<"<<Your Balance amount is: "<<user.balace<<">>\n\n";
                                    break;
                                }
                                case 2:
                                {
                                    user.deposit();
                                    break;
                                }
                                case 3:
                                {
                                    user.withdraw();
                                    break;
                                }
                                case 4:
                                {
                                    user.details();
                                    break;
                                }
                                case 5:
                                {
                                    goto end;
                                }
                                default:
                                {
                                    cout<<"**Please enter valid number**\n";
                                }

                            }
                        }
                    
                    }while(1);

                }
                else
                {
                    cout<<"PASSWORD IS INCORRECT\n";
                }
        }while(1);
    }
    else
    {
        cout<<"Please enter valid number\n";
    }
    
    //added a comment 
    end:

     return 0;
}