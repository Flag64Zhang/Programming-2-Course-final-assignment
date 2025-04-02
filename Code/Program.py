import tkinter as tk
from tkinter import messagebox
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os
from PIL import Image, ImageTk
import shutil
import jieba
import matplotlib.pyplot as plt
from tkinter import filedialog
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SmartNoteOrganizer:
    def __init__(self, master):
        self.master = master
        self.master.title("智能笔记整理器")

        # 初始化NLTK
        nltk.download('punkt')
        nltk.download('stopwords')

        # 初始化分类器
        self.classifier = self.train_classifier()

        self.create_widgets()


    def create_widgets(self):
        # 加载背景图片并调整大小
        # 获取当前脚本的路径
        current_path = os.path.dirname(__file__)

        # 构建相对路径
        image_path = os.path.join(current_path, '1.jpg')

        # 加载图片并调整大小
        self.original_img = Image.open(image_path)
        self.resized_img = self.original_img.resize((800, 600), Image.LANCZOS)
        self.background_img = ImageTk.PhotoImage(self.resized_img)

        # 创建Canvas并插入背景图片
        self.canvas = tk.Canvas(self.master, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_img)
        
        # 更换背景按钮
        self.save_button = tk.Button(self.canvas, text="更换背景图", command=self.change_background,
                                     padx=10, pady=5, bg='lightblue')
        self.save_button.place(relx=0.4, rely=0.4, anchor=tk.CENTER)

        # 文本输入框
        self.text_input = tk.Text(self.canvas, height=10, width=50)
        self.text_input.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

        # 提取关键词按钮
        self.extract_button = tk.Button(self.canvas, text="提取关键词", command=self.extract_keywords,
                                        padx=10, pady=5, bg='lightblue')
        self.extract_button.place(relx=0.2, rely=0.4, anchor=tk.CENTER)

        # 识别文本按钮
        self.classify_button = tk.Button(self.canvas, text="识别文本", command=self.classify_text,
                                         padx=10, pady=5, bg='lightblue')
        self.classify_button.place(relx=0.6, rely=0.4, anchor=tk.CENTER)

        # 保存按钮
        self.save_button = tk.Button(self.canvas, text="保存", command=self.save_note,
                                     padx=10, pady=5, bg='lightblue')
        self.save_button.place(relx=0.7, rely=0.4, anchor=tk.CENTER)

        # 生成柱状图按钮
        self.plot_button = tk.Button(self.canvas, text="生成柱状图", command=self.generate_bar_chart,
                                     padx=10, pady=5, bg='lightblue')
        self.plot_button.place(relx=0.8, rely=0.4, anchor=tk.CENTER)

        # 创建空白的图表容器
        self.bar_chart_container = tk.Frame(self.canvas, width=400, height=150, bg='white')
        self.bar_chart_container.place(relx=0.5, rely=0.68, anchor=tk.CENTER)
        
    def change_background(self):
        # 打开文件对话框，让用户选择新的背景图
        new_background_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="选择新的背景图")
        
        # 如果用户选择了文件
        if new_background_path:
            # 创建原始背景图的备份
            backup_path = "backup/1_backup.jpg"
            shutil.copy("1.jpg", backup_path)
            
            # 将新背景图复制为原背景图的名称
            shutil.move(new_background_path, "1.jpg")
            messagebox.showinfo("更换成功", "下次重启生效！")
            
            # 更新背景图
            self.background_path = "1.jpg"
            self.background_image = Image.open(self.background_path)
            self.background_photo = ImageTk.PhotoImage(self.background_image)
            self.background_label.config(image=self.background_photo)
        else:
            messagebox.showwarning("更换失败","请确认文件是否正确")

    def save_note(self):
        note_text = self.text_input.get("1.0", tk.END).strip()
        if note_text:
            # 指定 Excel 文件保存路径
            excel_filename = "notes.xlsx"

            # 分类主题
            topic = self.classify_topic(note_text)

            # 将关键词转换为列表
            seg_list = list(jieba.cut(note_text))
            new_seg_list = [item for item in seg_list if item != "\n"]
            #删除虚词
            new_seg_list = [item for item in new_seg_list if item not in ['的', '了', '着', '得', '地', '和', '与', '呢','吗','吧','呀','哦','是','不是']]
            # 统计词频
            word_counts = {}
            for word in new_seg_list:
                if word.isalnum():
                    word_counts[word] = word_counts.get(word, 0) + 1

            # 排序词频并提取前3个词
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            top_words = sorted_words[:3]

            # 将结果转换为字符串
            top_words_str = ', '.join([word for word, freq in top_words])

            # 保存笔记到 Excel 文件
            note_df = pd.DataFrame({"Note": [note_text], "Topic": [topic], "Keywords": [top_words_str]})
            note_df.to_excel(excel_filename, index=False)
            messagebox.showinfo("成功", "笔记已保存到 Excel 文件中！")
        else:
            messagebox.showwarning("警告", "请输入笔记内容！")

    def extract_keywords(self):
        note_text = self.text_input.get("1.0", tk.END).strip()
        if note_text:
            # 使用jieba进行分词
            seg_list = jieba.cut(note_text)
            
            #删除虚词
            new_seg_list = [item for item in seg_list if item not in ['的', '了', '着', '得', '地', '和', '与', '呢','吗','吧','呀','哦','是','不是']]

            # 统计词频
            word_counts = {}
            for word in new_seg_list:
                if word.isalnum():
                    word_counts[word] = word_counts.get(word, 0) + 1

            # 排序词频并提取前3个词
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            top_words = sorted_words[:3]
             

            # 将结果转换为字符串
            top_words_str = ', '.join([word for word, freq in top_words])

            # 显示关键字信息框
            messagebox.showinfo("关键词", "提取的关键词：" + top_words_str)
        else:
            # 提示用户输入内容
            messagebox.showwarning("警告", "请输入笔记内容！")

    def classify_text(self):
        note_text = self.text_input.get("1.0", tk.END).strip()
        if note_text:
            # 使用分类器预测主题
            topic = self.classify_topic(note_text)
            messagebox.showinfo("分类结果", "文本主题为：" + topic)
        else:
            messagebox.showwarning("警告", "请输入笔记内容！")

    def generate_bar_chart(self):
        note_text = self.text_input.get("1.0", tk.END).strip()
        if note_text:
            # 使用jieba进行分词
            seg_list = jieba.cut(note_text)
            #删除虚词
            new_seg_list = [item for item in seg_list if item not in ['的', '了', '着', '得', '地', '和', '与', '呢','吗','吧','呀','哦','是','不是']]
            # 统计词频
            word_counts = {}
            for word in new_seg_list:
                if word.isalnum():
                    word_counts[word] = word_counts.get(word, 0) + 1

            # 创建柱状图
            plt.figure(figsize=(4, 3))
            plt.bar(word_counts.keys(), word_counts.values())
            plt.xlabel('词语')
            plt.ylabel('词频')
            # # 在图表上方添加标题
            # plt.text(0.5, 1.05, '词频分析柱状图', horizontalalignment='center', verticalalignment='center',
            #          transform=plt.gca().transAxes)

            # 指定字体为SimHei，用于支持中文显示
            plt.xticks(fontproperties=fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf'), rotation=45, ha='right')
            plt.tight_layout()

            # 清空之前的图表容器并插入新生成的图表
            self.bar_chart_container.destroy()
            self.bar_chart_container = tk.Frame(self.canvas, width=400, height=150, bg='white')
            self.bar_chart_container.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.bar_chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            # 提示用户输入内容
            messagebox.showwarning("警告", "请输入笔记内容！")

    def train_classifier(self):
        # 示例数据集
        data = {
            "text": ["在爱里，你为何会颤抖呢","中国特色社会主义现代化强国",
                     "晴天雨天阴天下雨天多云","战争冲突坦克飞机","我通过炒股赚了很多钱",
                     "火箭汽车高铁人工智能AI大数据","高考中考考研","养生吃午饭晚饭早饭"],
            "label": ["爱情","政治","天气","军事","理财","科技","教育","健康"]
        }

        # 转换为 DataFrame
        df = pd.DataFrame(data)

        # 创建分类器
        classifier = make_pipeline(
            TfidfVectorizer(),
            MultinomialNB()
        )

        # 训练分类器
        classifier.fit(df['text'], df['label'])

        return classifier

    def classify_topic(self, text):
        # 使用分类器预测主题
        predicted_topic = self.classifier.predict([text])[0]
        return predicted_topic

def main():
    root = tk.Tk()
    app = SmartNoteOrganizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
