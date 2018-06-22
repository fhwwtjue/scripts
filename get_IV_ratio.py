import requests
import datetime
tradecal_url = "http://192.168.2.120/WEB/source/TradeCal"
import pickle
import time
strtradeday = list(requests.get(tradecal_url).json().values())

db_address = 'http://192.168.2.120:8086/write?db=monitor'

def write_data(ratio):
    send_data = '50_IV_Ratio value={} {:.0f}'.format(ratio, time.time() * 1e9)
    r = requests.post(db_address, data=send_data)
    if r.status_code!=204:
        print(r.status_code,r.text)

def is_tradingtime(tradetime,tradeday):
    if tradetime.strftime("%Y%m%d") not in tradeday:
        return False
    cou=tradetime.hour*60+tradetime.minute
    if (cou>=9*60+30 and cou<=11*60+30) or (cou>=13*60 and cou<=15*60):
        return True
    else:
        return False
def get_50():
    url="http://hq.sinajs.cn/list=sh510050"
    r=requests.get(url).text
    return float(r.split(',')[3])
def get_contractMonth():
    url='http://stock.finance.sina.com.cn/futures/api/openapi.php/StockOptionService.getStockName'
    r=requests.get(url)
    contractlist=r.json()['result']['data']['contractMonth']
    url2="http://stock.finance.sina.com.cn/futures/api/openapi.php/StockOptionService.getRemainderDay?date="
    if requests.get(url2+contractlist[0].replace("-","" )).json()['result']['data']['remainderDays'] >=10:
        return contractlist[0].replace("-","" )[2:]
    else:
        return contractlist[2].replace("-","" )[2:]
def get_optionlist():
    contractMonth=get_contractMonth()
    callurl='http://hq.sinajs.cn/list=OP_UP_510050'+contractMonth
    puturl='http://hq.sinajs.cn/list=OP_DOWN_510050'+contractMonth
    call=requests.get(callurl).text.split(',')[1:-1]
    put=requests.get(puturl).text.split(',')[1:-1]
    return [i[-8:] for i in call ],[i[-8:] for i in put ]

def get_option(contract,IV=False):
    url="http://hq.sinajs.cn/list=CON_SO_"+contract
    res=requests.get(url).text.split(',')
    if IV:
        return float(res[-8])
    else:
        return float(res[-4])
def get_atm():
    call={}
    put={}
    calllist,putlist=get_optionlist()
    price=get_50()
    for i in calllist:
        call[i]=get_option(i)
    for j in putlist:
        put[j]=get_option(j)
    atmcall=sorted(call.items(),key=lambda x:abs(x[1]-price))[:2]
    atmput=sorted(put.items(),key=lambda x:abs(x[1]-price))[:2]
    return atmcall,atmput

def get_ratio(call,put):
    IV1=get_option(call[0][0],True)
    IV2=get_option(call[1][0],True)
    IV3=get_option(put[0][0],True)
    IV4=get_option(put[1][0],True)
    return (IV1+IV2)/(IV3+IV4)

def is_after(tradeday):
    now=datetime.datetime.now()
    if now.strftime("%Y%m%d") not in tradeday:
        return False
    if now.hour>=15:
        return True
    else:
        return False


if __name__ == '__main__':
    atmcall,atmput=get_atm()
    lastdate=datetime.date(2018, 1, 17)
    # IVdict={}

    while(True):
        if is_tradingtime(datetime.datetime.now(),strtradeday) :
            ratio=get_ratio(atmcall,atmput)
            # IVdict[datetime.datetime.now()]=ratio
            write_data(ratio)
            # 收盘后更新一次
        elif is_after(strtradeday) and lastdate!=datetime.date.today():
            atmcall,atmput=get_atm()
            lastdate=datetime.date.today()
            # ratio = get_ratio(atmcall, atmput)
            # write_data(ratio)
        #     filename="IV_ratio_"+lastdate.strftime("%Y%m%d")
        #     with open(filename,'wb') as f:
        #         pickle.dump(IVdict,f)
        time.sleep(120) 
        # break  
