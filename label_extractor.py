import xml.etree.ElementTree as ET
import re

class BaseLabelExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def get_labels(self, data):
        ans = []
        for f in data:
            ans.append(self.extract_label(f))
        return ans

class ProblemExtractor(BaseLabelExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_label(self, item):
        r = r'(.+)-(.+)-(\d+).*'
        m = re.search(r, item)
        return m.group(2)

class VerdictExtractor(BaseLabelExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xml = kwargs.get("xml", "")
        self.root = ET.parse(self.xml).getroot()
        self.teams = {}
        for session in self.root[0][1:]:
            #print(session.attrib['alias'])
            tasks = []
            for problem in session:
                task = []
                for solution in problem:
                    task.append(solution.attrib['accepted'])    
                tasks.append(task)
            self.teams[session.attrib["alias"]] = tasks    


    def extract_label(self, item):
        r = r'(.+)-(.+)-(\d+)\..*'
        m = re.search(r, item)
        print(item)
        print(m.group(1))
        print(m.group(2))
        print(m.group(3))
        print(self.teams[m.group(1)])
        print(self.teams[m.group(1)][ord(m.group(2))-ord('a')])
        print(self.teams[m.group(1)][ord(m.group(2))-ord('a')][int(m.group(3)) - 1])
        print('-'*40)
        return self.teams[m.group(1)][ord(m.group(2))-ord('a')][int(m.group(3)) - 1]