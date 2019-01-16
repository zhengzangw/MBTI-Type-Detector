import tkinter as tk

class Application(tk.Frame):              
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)  
        self.grid()                       
        self.createWidgets()

    # 获取全局最后一个用户输入字符的位置 (坐标x, y)
    def __getLastAxis(self):
        lst = self.Text.index(tk.INSERT).split('.')
        return int(lst[0]), int(lst[1])

    def __axis2index(self, x, y):
        return str(x) + '.' + str(y)

    # 获取全局用户最后输入的最后一个字符的index("x.y")
    def __getLastIndex(self):
        x, y = self.__getLastAxis()
        return self.__axis2index(x, y)

    # Refresh Button 的回调函数
    def __refresh(self):
        total = self.Text.dump("1.0", self.Text.index(tk.END))

        text = ''
        for content in total:
            if content[0] == 'text':
                text += content[1]

        self.Text.tag_delete(self.Text.tag_names())
        self.Text.tag_add('a', '1.0', self.__getLastIndex())
        self.Text.tag_config('a', foreground = '#00ff00')

        print(text)

    # 键盘按键回调, 用于调试
    def __keyPressHandler(self, val):
        print(self.Text.index(tk.INSERT))
        print(val.char)
        pass

    def initText(self):
        self.Text = tk.Text(self)
        self.Text.grid(row=3, rowspan=2, column = 3, columnspan=3)
        self.Text.bind('<KeyPress>', self.__keyPressHandler)

    def initRefreshButton(self):
        self.refreshButton = tk.Button(self, text='Refresh', command=self.__refresh)
        self.refreshButton.grid()

    def createWidgets(self):
        self.initText()
        self.initRefreshButton()


if __name__ == '__main__':
    app = Application()                     
    app.master.title('Sample application')   
    app.mainloop()         
