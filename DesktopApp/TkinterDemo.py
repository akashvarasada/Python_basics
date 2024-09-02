# desktop application
# GUI
import sqlite3

from tkinter import *
import tkinter.messagebox as msg

def create_conn():
    return sqlite3.connect('DesktopApp/Database/python_mwf_10.db')

print(create_conn())

def insert_data():
    if e_fname.get() == "" or e_lname.get() == "" or e_email.get() == "" or e_mobile.get() == "":
        msg.showinfo("Insert status","All fields are necessary")
    else:
        conn=create_conn()
        cursor = conn.cursor()
        query = "insert into student(fname,lname,email,mobile) values(?,?,?,?)"
        args=(e_fname.get(),e_lname.get(),e_email.get(),e_mobile.get())
        cursor.execute(query,args)
        e_fname.delete(0,"end")
        e_lname.delete(0,"end")
        e_email.delete(0,"end")
        e_mobile.delete(0,"end")
        conn.commit()
        conn.close()


root = Tk()
root.geometry("500x500")

label = Label(root, text="My Tkinter Demo")
label.pack()

l_id = Label(root, text="ID")
l_id.place(x=50,y=50)

l_fname = Label(root, text="FIRST NAME")
l_fname.place(x=50,y=100)

l_lname = Label(root, text="LAST NAME")
l_lname.place(x=50,y=150)

l_email = Label(root, text="EMAIL")
l_email.place(x=50,y=200)

l_mobile = Label(root, text="MOBILE")
l_mobile.place(x=50,y=250)

e_id = Entry(root)
e_id.place(x=150,y=50)

e_fname = Entry(root)
e_fname.place(x=150,y=100)

e_lname = Entry(root)
e_lname.place(x=150,y=150)

e_email = Entry(root)
e_email.place(x=150,y=200)

e_mobile = Entry(root)
e_mobile.place(x=150,y=250)

insert_button = Button(root,text="INSERT",bg="black",fg="white",font=("Arial",12),command=insert_data)
insert_button.place(x=50,y=300)
root.mainloop()
