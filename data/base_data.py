# coding: utf-8

cn_dict = {
    '<bos>': 0,
    '<eos>': 1,
    '<pad>': 2,
    '是': 3,
    '千': 4,
    '你': 5,
    '万': 6,
    '在': 7,
    '我': 8,
    '人': 9,
    '三': 10,
    '一': 11,
    '侠': 12,
    '遍': 13,
    '二': 14,
    '爱': 15,
    '好': 16,
    '钢铁': 17
}

en_dict = {
    '<bos>': 0,
    '<eos>': 1,
    '<pad>': 2,
    'i': 3,
    'three': 4,
    'am': 5,
    'love': 6,
    'you': 7,
    'he': 8,
    'times': 9,
    'is': 10,
    'thousand': 11,
    'hello': 12,
    'iron': 13,
    'man': 14
}

sentence_demo = [
    [
        ('<bos>', 'i', 'am', 'iron', 'man', '<eos>'),
        ('<bos>', '我', '是', '钢铁', '侠', '<eos>')
    ]
]

sentence_pair_demo = [
    [
        ('<bos>', 'i', 'am', 'iron', 'man', '<eos>'),
        ('<bos>', '我', '是', '钢铁', '侠', '<eos>')
    ],
    [
        ('<bos>', 'i', 'love', 'you', 'three', 'thousand', 'times', '<eos>'),
        ('<bos>', '我', '爱', '你', '三', '千', '遍', '<eos>')
    ]
]
