import step1 as s1
import step2 as s2
import step3 as s3

technic_stocks = ['AAPL', 'AMZN', 'MSFT', 'CSCO', 'GOOG', 'FB']
mixed_stocks = ['AMZN', 'JNJ', 'AEP', 'HSBC', 'NGG', 'BA']

def main():
    for s in technic_stocks:
        s1.main(s)
        s2.main(s)
        s3.main(s)
    
if __name__ == '__main__':
    main()