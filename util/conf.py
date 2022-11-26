# class OptionConf(object):
#     def __init__(self,content):
#         self.line = content.strip().split(' ')
#         self.options = {}
#         self.mainOption = False
#         if self.line[0] == 'on':
#             self.mainOption = True
#         elif self.line[0] == 'off':
#             self.mainOption = False
#         for i,item in enumerate(self.line):
#             if (item.startswith('-') or item.startswith('--')) and  not item[1:].isdigit():
#                 ind = i+1
#                 for j,sub in enumerate(self.line[ind:]):
#                     if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
#                         ind = j
#                         break
#                     if j == len(self.line[ind:])-1:
#                         ind=j+1
#                         break
#                 try:
#                     self.options[item] = ' '.join(self.line[i+1:i+1+ind])
#                 except IndexError:
#                     self.options[item] = 1
#     def __getitem__(self, item):
#         if not self.contain(item):
#             print('parameter '+item+' is invalid!')
#             exit(-1)
#         return self.options[item]
#     def keys(self):
#         return self.options.keys()
#     def is_main_on(self):
#         return self.mainOption
#     def contain(self,key):
#         return key in self.options


