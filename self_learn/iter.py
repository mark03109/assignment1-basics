class Myclass:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return x

class MyListIterator():
    def __init__(self, lst):
        self.lst = lst
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.lst is None:
            raise StopIteration
        elif self.index >= len(self.lst):
            self.lst = None
            raise StopIteration 
        else:
            x = self.lst[self.index]
            self.index += 1
            return x 

class MyList(list):
    def __iter__(self):
        return MyListIterator(self)

def test1():
    myiter = Myclass() # 这里myiter 类自己就是迭代器了
    myiter = iter(myiter)

    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))


    myiter = iter(myiter)
    print(next(myiter))
    print(next(myiter))

def test2():
    lst = [1, 23, 4.3, 23, 232, 1, 2, 5]
    myList = MyList(lst)
    myiter = iter(myList)

    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))
    print(next(myiter))

if __name__ == "__main__":
    test2()