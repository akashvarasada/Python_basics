

class Bankcls:

    def openaccount(self,cname,acno,balance):
        self.customername = cname
        self.accountnumber = acno
        self.balance = balance
        print("Hello Welcome ",self.customername,". Your account has been opened with ",self.balance,"INR")

    def deposit(self,depositamount):
        self.balance = self.balance + depositamount
        print(depositamount," has been deposited into you account ",self.accountnumber)
        print("Your account balance is ",self.balance)

    def withdraw(self,withdrawalamount):
        if withdrawalamount <= self.balance:
            self.balance = self.balance - withdrawalamount
            print(withdrawalamount," has been debited from you account ",self.accountnumber)
            print("Your account balance is ",self.balance)
        else:
            print("Sorry insufficient balance.")

    def checkbalance(self):
        print("Your account balance is ",self.balance)
