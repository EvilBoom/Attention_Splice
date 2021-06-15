class Car:
    def __init__(self, name):
        self.name = name

    def price(self):
        pass


class Bieke1(Car):
    def __init__(self, name, days):
        super().__init__(name)
        self.days = days

    def price(self):
        print("租用%s的天数%d,价格为%d" % (self.name, self.days, 600 * self.days))


class Bmw(Car):
    def __init__(self, name, days):
        super().__init__(name)
        self.days = days

    def price(self):
        print("租用%s的天数%d,价格为%d" % (self.name, self.days, 500 * self.days))


class bieke2(Car):
    def __init__(self, name, days):
        super().__init__(name=name)
        self.days = days

    def price(self):
        print("租用%s的天数%d,价格为%d" % (self.name, self.days, 300 * self.days))


class Keche1(Car):
    def __init__(self, name, days):
        super().__init__(name)
        self.days = days

    def price(self):
        print("租用%s的天数%d,价格为%d" % (self.name, self.days, 800 * self.days))


class keche2(Car):
    def __init__(self, name, days):
        super().__init__(name)
        self.days = days

    def price(self):
        print("租用%s的天数%d,价格为%d" % (self.name, self.days, 1500 * self.days))


def main():
    while (1):
        print("请选择车型：1.别克商务舱GL8，600元一天 2.宝马550i, 500一天 3.别克林荫大道，300元一天 4.客车<=16座，800元一天 5.客车>16座，1500元一天")
        sizecar = int(input("车的类型1/2/3/4/5:"))
        days = int(input("使用天数："))

        if sizecar == 1:
            c = Bieke1("别克商务舱GLB", days)
            c.price()
        elif sizecar == 2:
            c = Bmw("宝马550i", days)
            c.price()
        elif sizecar == 3:
            c = bieke2("别克林荫大道", days)
            c.price()
        elif sizecar == 4:
            c = Keche1("客车<=16座", days)
            c.price()
        elif sizecar == 5:
            c = keche2("客车>16座", days)
        else:
            print("输出有误")
            continue


if __name__ == "__main__":
    main()
