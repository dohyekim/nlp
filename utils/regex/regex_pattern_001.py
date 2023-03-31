import re

def pattern_01():
    text = 'test-20200729'
    pattern = '[A-z]+'
    res = re.compile(pattern).findall(text)
    print(res)

def pattern_02():
    text = "건축 / 목공 ( ○명 )"
    pattern = '[\w+]\s+(\(\s+[0,O,○]명\s+\))'

    res = re.compile(pattern).findall(text)
    print(res)

    replaced = re.sub(pattern, "", text)
    print(replaced)
