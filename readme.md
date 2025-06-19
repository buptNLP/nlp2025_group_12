jupyter文件均为模型训练的文件，共四个版本。
inference_appultimate.py文件为web评测端文件

下面是数据集的介绍：
img文件夹下存放每一条声明的图片

img_html_news文件夹下存放根据每一条声明的caption检索到的网页与图片，其中direct_annotation.json包含如下信息：

~~~json
{
      "img_link": 检索到的相关图片的链接,
      "page_link": 检索到的网页链接,
      "domain": 检索到的网页的域名,
      "snippet": 检索到的网页的简洁摘要,
      "image_path": 检索到的图片的路径,
      "html_path": 检索到的网页的路径,
      "page_title": 检索到的网页标题
}
~~~

inverse_search文件夹下存放根据声明的图片找到的网页，其中inverse_annotation.json包含如下信息

~~~json
{
"entities": 声明中图片中的实体, 
"entities_scores": 声明中图片中的实体的分数, 
"best_guess_lbl": 声明中图片最可能是什么, 
"all_fully_matched_captions": , 
"all_partially_matched_captions":
"fully_matched_no_text": 
上述三个字段的值均为寻找到的网页，为一个列表，列表中的元素为一个字典，格式如下
	{
	"page_link": 检索到的网页链接, 
	"image_link": 检索到的图片链接, 
	"html_path": 检索到的网页的路径, 
	"title": 检索到的网页的标题
	}
}

~~~

