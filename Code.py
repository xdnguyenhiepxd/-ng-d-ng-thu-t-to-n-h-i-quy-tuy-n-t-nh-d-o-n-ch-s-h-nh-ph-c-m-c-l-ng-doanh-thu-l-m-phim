import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
from sklearn.metrics import mean_squared_error, r2_score #các hàm đánh giá
from sklearn.model_selection import train_test_split

data=""
X=""
Y=""
TitleX=""
TitleY=""
X_p=""
Y_y=""

print("<-------------Chọn dữ liệu:------------->")
print("1. income.data.csv")
print("2. Salary_data.csv")
print("3. cost_revenue_clean.csv")
while 1==1:
	name_data = input()
	if( 0 < int(name_data) < 4 ) :
		select_number = int(name_data)
		if select_number == 1 :
			data ="income.data.csv"
			X="income"
			Y="happiness"
			TitleX="Chỉ số thu nhập"
			TitleY="Chỉ số hạnh phúc"
			X_p="thu nhập"
			Y_p="hạnh phúc"
			break
		elif select_number == 2 :
			data ="Salary_data.csv"
			X="Experience"
			Y="Salary"
			TitleX="Kinh nghiệm"
			TitleY="Mức lương"
			X_p="Kinh nghiệm"
			Y_p="Mức lương"
			break
		elif select_number == 3 :
			data ="cost_revenue_clean.csv"
			X="production_budget_usd"
			Y="worldwide_gross_usd"
			TitleX="Kinh phí"
			TitleY="Doanh thu toàn thế giới"
			X_p="kinh phí"
			Y_p="doanh thu"
			break
	else :
		print("Yêu cầu nhập đúng dữ liệu!")

income = pd.read_csv(data)
x_income=income[X].values
y_income=income[Y].values
print("==============================================")
print("X là:")
print(x_income)
print("==============================================")
print("Y là:")
print(y_income)
print("==============================================")
print("X max:" +str(x_income.max()))
print("==============================================")
print("X min:" +str(x_income.min()))
print("==============================================")
print("Y max:" +str(y_income.max()))
print("==============================================")
print("Y min:" +str(y_income.min()))
print("==============================================")
gia_tri_tb_X=statistics.mean(x_income)
gia_tri_tb_Y=statistics.mean(y_income)
print("Giá trị trung bình của X là:" + str(gia_tri_tb_X))
print("==============================================")
print("Giá trị trung bình của Y là:" + str(gia_tri_tb_Y))
print("==============================================")
print("Phương sai là của X là: " + str(np.var(x_income)))
print("==============================================")
print("Độ lệch chuẩn của X là: " + str(np.std(x_income)))
print("==============================================")
print("Phương sai là của Y là: " + str(np.var(y_income)))
print("==============================================")
print("Độ lệch chuẩn của Y là: " + str(np.std(y_income)))
print("==============================================")
X_train, X_test, y_train, y_test = train_test_split(x_income, y_income, test_size=0.5, random_state=35)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

x_income_convert = x_income.reshape(x_income.shape[0],-1)#convert
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor

regr = linear_model.LinearRegression().fit(x_income_convert, y_income)
# regr = DecisionTreeRegressor(random_state = 0)
# regr.fit(x_income_convert, y_income)
print("Nhập chỉ số "+ X_p + " cần dự đoán:")
X_input = input()
print ('Dự đoán chỉ số '+ Y_p +' là: ', regr.predict([[float(X_input)]]))

x_income_test = X_test.reshape(X_test.shape[0],-1)#convert
y_pred = regr.predict(x_income_test)

w0 = regr.intercept_
w1 = regr.coef_
print ('w0 = ', w1)
print( 'w1 : ', w0)
#print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))
# #tương tự đường thằng y = ax + b
# # a = w1, b = w0

#X_train_convert = X_train.reshape(X_train.shape[0],-1)
#plt.scatter(x_income, y_income, color='red')
plt.scatter(X_train, y_train, color='green')
#plt.scatter(X_train, regr.predict(X_train_convert), color='red')
plt.axis([x_income.min()*0.5, x_income.max()*1.2, y_income.min()*0.5, y_income.max()*1.5])
plt.xlabel(TitleX)
plt.ylabel(TitleY)
# #Vẽ đường thẳng với các hệ số huấn luyện được
x0 = np.linspace(x_income.max()*2, x_income.min(),int(x_income.max()/x_income.min()))
y0 = w0 + w1*x0
plt.plot(x0, y0)
plt.show()


