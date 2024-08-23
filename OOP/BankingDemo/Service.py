import Bank


def servicedemo(cname,acno,balance):

    b1 = Bank.Bankcls()
    b1.openaccount(cname,acno,balance)

    while True:
        print("*"*30)
        print("Welcome to ABC Bank")
        print("Select any of the below services by entering the index number.")
        print("1. Deposit")
        print("2. Withdraw")
        print("3. Check Balance")
        print("4. Exit")
        print("*"*30)

        choice = int(input("Enter the service number."))
        if choice == 1:
            depositamount = int(input("Enter the deposit amount : "))
            b1.deposit(depositamount)
            print("*"*30)
        elif choice == 2:
            withdrawalamount = int(input("Enter the withdrawal amount: "))
            b1.withdraw(withdrawalamount)
            print("*"*30)
        elif choice == 3:
            b1.checkbalance()
            print("*"*30)
        elif choice == 4:
            print("Thank you for using the ABC banking services.")
            print("*"*30)
            break
        else:
            print("Invalid Choice, Please try again")
