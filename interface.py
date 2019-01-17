import tkinter as tk
from PIL import Image, ImageTk
#import demo as vis
import randomColor as vis

# 用于16进制转换
Dec2Hex = '0123456789abcdef'

# 用于从性格数组得到图片名
PST = [['I', 'E'],
       ['N', 'S'],
       ['T', 'F'],
       ['J', 'P']]

class Application(tk.Frame):              
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)  
        self.grid()                       
        self.createWidgets()
        self.classType = 0 # 人格分类坐标
        self.tagCounter = 0 # tag名字计数

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

    # 获取Text中的文本
    def __dumpText(self):
        total = self.Text.dump("1.0", self.Text.index(tk.END))
        text = ''
        for content in total:
            if content[0] == 'text':
                text += content[1]
        text = text[:-1] # 去掉最后一个'\n'
        return text
    
    # (重新)计算当前sentence的颜色
    # (重新)计算当前用户性格
    def __calcColorDict(self):
        text = self.__dumpText()
        self.color_dict, personality = vis.gen_color(text)
        self.__updatePersonalityPic(personality)
        return self.color_dict

    # 将R,G, B颜色转化为16进制字符串表示
    def __convertHexColor(self, color):
        color = [min(255, max(0, c)) for c in color] # 防止输入的颜色越界
        f = lambda c: Dec2Hex[c // 16]+ Dec2Hex[c % 16]
        return '#' + f(color[0]) + f(color[1]) + f(color[2])

    # 向Text中输入一个字(可以带颜色 'str', [R, G, B])
    def __insertWord(self, word, color = None):
        pt = self.Text.index(tk.INSERT)
        self.Text.insert(self.Text.index(tk.INSERT), word)
        if color != None:
            self.tagCounter += 1
            self.Text.tag_add(str(self.tagCounter), pt, self.Text.index(tk.INSERT))
            self.Text.tag_config(str(self.tagCounter), foreground = self.__convertHexColor(color))

    # 根据字典在Text中高亮并显示句子
    def parseSentence(self):
        seq = self.color_dict[self.classType]
        if len(seq) > 0:
            self.__insertWord(seq[0][0], seq[0][1])
        seq = seq[1:]
        for item in seq:
            self.__insertWord(' ')
            self.__insertWord(item[0], item[1])

    def __deleteAllTags(self):
        self.tagCounter = 0
        self.Text.tag_delete(self.Text.tag_names())
        self.Text.delete('1.0', self.Text.index(tk.END))

    # Refresh Button 的回调函数
    def refresh(self):
        # 重新计算颜色
        self.__calcColorDict()
        # 清空buffer
        self.__deleteAllTags()
        # 重新显示文本 
        self.parseSentence()

    # 键盘按键回调, 用于调试
    def __keyPressHandler(self, val):
        if val.char == ' ':
            self.refresh()

    def initText(self):
        self.Text = tk.Text(self)
        self.Text.grid(row=0, rowspan = 30, column = 1, columnspan = 30)
        self.Text.bind('<KeyPress>', self.__keyPressHandler)

    def initRefreshButton(self):
        self.refreshButton = tk.Button(self, text='Refresh', command=self.refresh)
        self.refreshButton.grid(column = 1, row = 31)

    def __IESelected(self):
        self.classType = 0
    def __NSSelected(self):
        self.classType = 1
    def __TFSelected(self):
        self.classType = 2
    def __JPSelected(self):
        self.classType = 3

    # 初始化四分类的选择按钮
    def initClassButton(self):
        self.classButtonList = []
        cate = ['IE', 'NS', 'TF', 'JP']
        func = [self.__IESelected, self.__NSSelected, self.__TFSelected, self.__JPSelected]
        for i in range(4):
            self.classButtonList.append(tk.Button(self, text = cate[i], command = func[i]))
            self.classButtonList[i].grid(row = i, rowspan = 1, column = 0, columnspan = 1)

    # 自适应图片大小
    def __adaptImage(self, image, x, y):
        _x, _y = image.size
        rx = x / _x
        ry = y / _y
        max_r = max(rx, ry)
        min_r = min(rx, ry)
        r = 0
        if min_r < 1: 
            r = min_r
        else: 
            r = max_r
        x_new = int(_x * r)
        y_new = int(_y * r)
        return image.resize((x_new, y_new))

    # 初始化人格类型图片
    def initPersonalityPic(self):
        self.image = Image.open('pic/INTP.png')
        self.image = self.__adaptImage(self.image, 400, 600)
        self.personalityImage = ImageTk.PhotoImage(self.image)
        self.Canvas = tk.Canvas(width = 400, height=600)
        self.imageID = self.Canvas.create_image(200, 300, image = self.personalityImage)
        self.Canvas.grid(column = 31, row = 0)

    # 根据personality = [0, 1, 0, 1]来更新当前的性格图片
    def __updatePersonalityPic(self, personality):
        fname = 'pic/'
        for i in range(4):
            fname += PST[i][personality[i]]
        fname += '.png'
        self.image = Image.open(fname)
        self.image = self.__adaptImage(self.image, 400, 600)
        self.personalityImage = ImageTk.PhotoImage(self.image)
        self.Canvas.itemconfig(self.imageID, image = self.personalityImage)

    # 生成word graph
    def wordGraph(self):
        self.wordGraphFrame = tk.Toplevel(self)
        self.wordGraphCanvas = tk.Canvas(master = self.wordGraphFrame, width = 800, height = 600)
        self.wordGraphImage_PIL = vis.get_wordcloud(self.__dumpText())
        self.wordGraphImage_PIL = self.__adaptImage(self.wordGraphImage_PIL, 800, 600)
        self.wordGraphImage = ImageTk.PhotoImage(self.wordGraphImage_PIL)
        self.wordGraphImageID = self.wordGraphCanvas.create_image(400, 300, image = self.wordGraphImage)
        self.wordGraphCanvas.grid()

    def initWordGraphButton(self):
        self.wordGraphButton = tk.Button(self, text = 'word graph', command = self.wordGraph)
        self.wordGraphButton.grid(column = 2, row = 31)

    def createWidgets(self):
        self.initText()
        self.initRefreshButton()
        self.initClassButton()
        self.initPersonalityPic()
        self.initWordGraphButton()


if __name__ == '__main__':
    vis.load_model_and_data("cnn16seq.h5")
    app = Application()                     
    app.master.title('Sample application')   
    app.mainloop()         
