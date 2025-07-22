"""
ll
2025.3.30
爬取静态页面中多个子页面的信息
"""
import requests
from bs4 import BeautifulSoup
import time

# 基础URL（用于拼接相对链接）
base_url = "https://www.xyyuedu.com"

# 1. 获取章节列表页
list_url = "https://www.xyyuedu.com/gudaiyishu/huangdinajingbaihuawen/"  # 替换为实际URL
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# 获取章节链接
def get_chapter_links(url):
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到包含所有链接的ul元素
    ul = soup.find('ul', class_='zhangjie2')
    if not ul:
        print("未找到章节列表！")
        return []

    # 提取所有a标签的href
    links = [a['href'] for a in ul.find_all('a', href=True)]
    return links

def get_chapter_content(chapter_url):
    full_url = base_url + chapter_url if chapter_url.startswith('/') else chapter_url
    try:
        response = requests.get(full_url, headers=headers)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')

        # 找到目标div
        target_div = soup.find('div', {'id': 'onearcxsbd', 'class': 'onearcxsbd'})
        if not target_div:
            print(f"未找到内容div：{full_url}")
            return ""
        # print(target_div)
        # 提取从div开始到<!--分页-->之前的内容
        content = []
        for element in target_div.next_elements:
            if str(element).strip() == '分页':
                break
            if isinstance(element, str):
                content.append(element.strip())
            else:
                content.append(element.get_text().strip())
        print(content)
        return "".join(content).strip()
    except Exception as e:
        print(f"获取章节内容出错：{full_url}, 错误：{e}")
        return ""
def main():
    # 获取所有章节链接
    chapter_links = get_chapter_links(list_url)
    if not chapter_links:
        print("没有获取到章节链接！")
        return

    print(f"共找到 {len(chapter_links)} 个章节")

    # 存储所有章节内容
    all_contents = []

    # 遍历每个章节
    for i, link in enumerate(chapter_links, 1):
        print(f"正在处理第 {i} 章: {link}")
        content = get_chapter_content(link)
        if content:
            all_contents.append(content)
        time.sleep(1)  # 礼貌爬取，避免给服务器造成压力

    # 4. 保存到txt文件
    with open("huangdi_data.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_contents))  # 两个换行符分隔不同章节

    print("所有章节已保存到 huangdi_data.txt")


if __name__ == "__main__":
    main()