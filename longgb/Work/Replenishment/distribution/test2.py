__author__ = 'gaoyun3'
from datetime import *
def main():
    dt = datetime.today().date() + timedelta(days=-1)


    print datetime.strftime(datetime.today().date() + timedelta(days=-1),"%Y-%m-%d")
    todayDate = datetime.today()
    beginYearDate = "'"+todayDate.replace(year=(todayDate.year-1)).strftime("%Y-%m-%d")+"'"
    print beginYearDate

if __name__ == '__main__':
    main()


