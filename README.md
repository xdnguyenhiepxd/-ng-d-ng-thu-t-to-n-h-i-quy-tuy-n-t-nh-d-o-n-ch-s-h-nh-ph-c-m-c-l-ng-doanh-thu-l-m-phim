# Sử dụng thuật toán Linear Regression để dự đoán chỉ số hạnh phúc của con người, dự đoán mức lương dựa vào kinh nghiệm, dự đoán doanh thu phim tại Box Office
## 1. Mô tả thuật toán
### 1.1 Giới thiệu mô hình Regression:
- "Regression" là một phương pháp thống kê để hồi quy dữ liệu với biến phụ thuộc có giá trị liên tục trong khi các biến độc lập có thể có một trong hai giá trị liên tục hoặc là giá trị phân loại. Nói cách khác "Regression" là một phương pháp để dự đoán biến phụ thuộc (Y) dựa trên giá trị của biến độc lập (X). Nó có thể được sử dụng cho các trường hợp chúng ta muốn dự đoán một số lượng liên tục. Ví dụ, dự đoán giao thông ở một cửa hàng bán lẻ, dự đoán thời gian người dùng dừng lại một trang nào đó hoặc số trang đã truy cập vào một website nào đó v.v...
- Mục tiêu của giải thuật hồi quy tuyến tính là dự đoán giá trị của một hoặc nhiều biến mục tiêu liên tục (continuous target variable) y dựa trên một véc tơ đầu vào X.
- Ví dụ: Dự đoán giá nhà ở Hà Nội dựa vào thông tin về diện tích, vị trí, năm xây dựng của ngôi nhà thì ở đây là giá nhà và X=(x1,x2,x3) với x1 là diện tích ,x2 là vị trí và x3 là năm xây dựng.
- Ví dụ về bài toán dự đoán giá của căn nhà ta có thể thấy rằng :
  - Diện tích căn nhà càng lớn=>giá càng cao.
  - Số lượng phòng lớn=>Giá càng cao.
  - Càng xa trung tâm =>Giá càng giảm.
- Đây là ví dụ về dự đoán phép toán:<br>
![vidupheptoan](https://user-images.githubusercontent.com/88564663/137233583-f9132c78-dd95-479a-ad87-f3281433349f.png)
- Trong vòng khoảng nửa phút ,bạn sẽ ra cách tìm dấu điền vào chỗ trống để có  được 1 kết quả đúng. Và Machine-Learning cũng vậy,bạn đưa cho máy tính con số và kết quả ,việc của máy tính là tìm ra mối  liên hệ giữa các con số để đồng nhất kết quả giữa vế trái và vế phải của phép tính. Về cơ  bản thì  ta sẽ có 1 cặp huấn luyện chứa các cặp (x(i),y(i)),tương ứng  với nhiệm vụ của ta là phải tìm ra giá trị ứng với 1 đầu vào X mới.
- Để làm được điều này ta cần phải tìm được quan hệ giữa x và y để từ đó đưa ra dự đoán.
![quanhey](https://user-images.githubusercontent.com/88564663/137234087-bfb64297-615a-4931-a85c-897a704d3612.png)
- Một hàm số đơn giản nhất có thể mô tả mối quan hệ giữa giá nhà và 3 đại lượng đầu vào là:
![hamy](https://user-images.githubusercontent.com/88564663/137234838-6b95b82c-7db2-4e08-9a14-68402f58fd93.png)
- Trong đó ,w1 ,w2 ,w3 ,w0  là các hằng số, w0  còn được gọi là bias.Mối quan hệ y=f(x) bên trên là một mối quan hệ tuyến tính (linear). Bài toán chúng ta đang làm là một bài toán thuộc loại regression. Bài toán đi tìm các hệ số tối ưu {w1,w2,w3,,w0} {w1,w2,w3,w0} chính vì vậy được gọi là bài toán Linear Regression.
### 1.2 Mô hình Linear Regression:
- Mô hình đơn giản nhất là mô hình kết hợp tuyến tính của các biến đầu vào:
![hamy2](https://user-images.githubusercontent.com/88564663/137234789-99316d86-83ef-4a24-9add-5275f699d6c0.png)
- Trong đó là véc tơ biến đầu vào và là véctơ trọng số tương ứng. Thường θ được gọi là tham số của mô hình. Giá trị của tham số sẽ được ước lượng bằng cách sử dụng các cặp giá trị của tập huấn luyện.
- Thực ra mô hình tuyến tính là chỉ cần ở mức tuyến tính giữa tham số  θ và y là đủ. Và mình cho rằng tên gọi tuyến tính là xuất phát giữa θ và y chứ không phải giữa x và y. Nói cách khác, ta có thể kết hợp các x một cách phi tuyến trước khi hợp với θ để được y. Một cách đơn giản là sử dụng hàm phi tuyến cho x như sau:
![hamy3](https://user-images.githubusercontent.com/88564663/137235007-15271379-34f2-4398-a9a6-ea85e3364245.png)
- θ được gọi là tham số của mô hình. Giá trị của tham số sẽ được ước lượng bằng cách sử dụng các cặp giá trị (x(i),y(i))  của tập huấn luyện.
- θo được gọi là độ lệch (bias) nhằm cắt đi mức độ chênh lệch giữa mô hình và thực tế.
- Viết lại công thức trên như sau: 
![hamy4](https://user-images.githubusercontent.com/88564663/137235174-334b1f0e-2e3c-4522-a323-c3e26554289b.png)
- Như quy ước thì tất cả các véc tơ  nếu không nói gì thì ta ngầm định với nhau rằng nó là véc tơ cột nên ta có được cách viết nhân ma trận như trên.
### 1.3 Sai số của dự đoán:
- Chúng ta mong muốn rằng sự sai khác e giữa giá trị thực y và giá trị dự đoán y^ (đọc là y hat trong tiếng Anh) là nhỏ nhất. Nói cách khác, chúng ta muốn giá trị sau đây càng nhỏ càng tốt:<br>
![hame](https://user-images.githubusercontent.com/88564663/137236109-0472340d-67dd-4af1-8e22-e3202cc9e389.png)
### 1.4 Xác định Basic Function:
![hinhcau](https://user-images.githubusercontent.com/88564663/137236211-5e6dc6fe-bb30-4857-9f86-49facba8509d.png)
- Giữ nguyên đầu vào có ý là không thay đổi giá trị đầu vào ϕi (x)=x.
- Chuẩn hoá về đoạn [min, max].<br>
![chuanhoa](https://user-images.githubusercontent.com/88564663/137236303-4a5c6888-73bf-414b-8aaf-f0cb9e8ac970.png)
- Sử dụng đa thức bậc cao:<br>
![chuanhoa1](https://user-images.githubusercontent.com/88564663/137236380-c7860172-db0b-4380-a1b0-0c0592baddda.png)
- Sử dụng hàm Gaussian: <br>
![chuanhoa2](https://user-images.githubusercontent.com/88564663/137236443-591353ca-6232-4a4b-8a4e-7870b1abc22c.png)
### 1.5 Hạn chế của mô hình Linear Regression:
- Hạn chế đầu tiên của Linear Regression là nó rất nhạy cảm với nhiễu (sensitive to noise). Trong ví dụ về mối quan hệ giữa chiều cao và cân nặng bên trên, nếu có chỉ một cặp dữ liệu nhiễu (150 cm, 90kg) thì kết quả sẽ sai khác đi rất nhiều. Xem hình dưới đây:<br> 
![giatri](https://user-images.githubusercontent.com/88564663/137236894-3a09bc16-24c8-45c7-97e2-e98f1ec6c06a.png)
<br>
-Đồ thị thể hiện giá trị hiện tại và giá trị tương lai của dữ liệu.<br><br>

- Vì vậy, trước khi thực hiện Linear Regression, các nhiễu (outlier) cần phải được loại bỏ. Bước này được gọi là tiền xử lý (pre-processing).
- Hạn chế thứ hai của Linear Regression là nó không biểu diễn được các mô hình phức tạp. Mặc dù trong phần trên, chúng ta thấy rằng phương pháp này có thể được áp dụng nếu quan hệ giữa outcome và input không nhất thiết phải là tuyến tính, nhưng mối quan hệ này vẫn đơn giản nhiều so với các mô hình thực tế.
### 1.6 Ứng dụng trong giáo dục:
- Dự báo là phán đoán những sự kiện sẽ xảy ra trong tương lai trên cơ sở phân tích khoa học các dữ liệu của quá khứ và hiện tại nhờ một số mô hình toán học.
- Dự báo Giáo dục là việc đưa ra các dự báo những sự kiện Giáo dục sẽ xảy ra trong tương lai dựa trên cơ sở phân tích khoa học các số liệu kinh tế của quá khứ và hiện tại. Chẳng hạn, nhà quản lý dựa trên cơ sở các số liệu về điểm thi đầu vào của kỳ trước và kỳ này để đưa ra dự báo về điểm tuyển sinh của các trường học trong  tương lai.
- Do đó, trong hoạt động Giáo dục, dự báo đem lại ý nghĩa rất lớn. Nó là cơ sở để lập các kế hoạch học tập tạo tính hiệu quả và sức cạnh tranh cho các sĩ tử trong tương lai. Dự báo mang tính khoa học và đòi hỏi cả một nghệ thuật dựa trên cơ sở phân tích khoa học các số liệu thu thập được. Bởi lẽ cũng dựa vào các số liệu thời gian nhưng lấy số lượng là bao nhiêu, mức độ ở những thời gian cuối nhiều hay ít sẽ khiến cho mô hình dự đoán phản ánh đầy đủ hay không đầy đủ những thay đổi của các nhân tố mới đối với sự biến động của hiện tượng. Do vậy mà dự báo vừa mang tính chủ quan vừa mang tính khách quan. Dự báo muốn chính xác thì càng cần phải loại trừ tính chủ quan của người dự báo.
## 2. Thử nghiệm và đánh giá kết quả
### 2.1 Phát biểu bài toán
- Bài toán dự đoán chỉ số hạnh phúc của con người đưa ra thông tin về chỉ số lương của con từ đó làm căn cứ dự chỉ số hạnh phúc.
- Bài toán sẽ lấy dữ liệu trên Kaggle để phân tích ,huấn luyện để dự đoán chỉ số hạnh phúc của con người.
- Bài toán dự đoán chỉ số lương của con người đưa ra thông tin về chỉ số kinh nghiệm làm việc tính theo năm của con người từ đó làm căn cứ dự  đoán chỉ số lương.
- Bài toán sẽ lấy dữ liệu trên Kaggle để phân tích ,huấn luyện để dự đoán chỉ số lương hàng năm của từng nhân viên.
- Bài toán dự đoán chỉ số doanh thu của bộ phim dựa trên kinh phí của bộ phim làm ra.
- Bài toán sẽ lấy dữ liệu trên Kaggle để phân tích ,huấn luyện để dự đoán chỉ số doanh thu của phim..
### 2.2 Chuẩn bị dữ liệu                      
![income](https://user-images.githubusercontent.com/88564663/137229135-c71c9a39-d9d2-41c4-af8d-465449af5d78.png)
<br>
-Dữ liệu file income.data.csv<br><br><br>
![Salary_data](https://user-images.githubusercontent.com/88564663/137231241-7d5946b9-3d8c-4a69-bb7f-6cfc362b9916.png)
<br>
-Dữ liệu file Salary_data.csv<br><br><br>
![file](https://user-images.githubusercontent.com/88564663/137231390-8eaa448a-9d23-4a0a-a434-ef2d4d5d9e54.png)
<br>
-Dữ liệu file cost_revenue_clean.csv<br>
### 2.3 Xử lý dữ liệu
- Ở đây chúng em sử dụng Linear regression và dữ liệu lấy từ Excel(file csv) ,subline text,python ,command prompt  để hỗ trợ quá trình training .Về cơ bản thì python đã được tích hợp rất nhiều các thuật toán khác nhau, dễ dàng sử dụng, và giúp giảm thời gian xây dựng các hệ thống deep learning. Đồng thời kết hợp với pandas và numpy để phân tích, và xử lý cấu trúc data, và matplotlib dùng để về đồ thị. 
- Việc vẽ đồ thị rất quan trọng đối với các bài toán thuộc dạng hồi quy tuyến tính như thế này. Vì dĩ nhiên việc đoán trước không thể trả về kết quả chính xác 100% được, Kết quả sẽ là tương đối và có thể có một chút sai số không đáng kể. Vì thế việc vẽ đồ thị sẽ giúp bạn dễ dàng so sánh giữa kết quả dự đoán và thực tế.
### 2.4 Chạy chương trình
#### 2.4.1 Dữ liệu income_data.csv
![cincome](https://user-images.githubusercontent.com/88564663/137232243-b8a58fd4-8594-4ace-92d6-29d32ba5c536.png)
<br>
-Nạp dữ liệu file income_data.csv<br><br><br>
![vincome](https://user-images.githubusercontent.com/88564663/137232479-ac5ec7b3-fc7c-4a46-863e-dfa466c92642.png)
<br>
-Mô hình dự đoán kết quả<br><br>
#### 2.4.2 Dữ liệu Salary_data.csv
![csalary](https://user-images.githubusercontent.com/88564663/137232974-582bdf7c-7fe1-462d-a1a7-30f87686a90e.png)
<br>
-Nạp dữ liệu file Salary_data.csv<br><br><br>
![vsalary](https://user-images.githubusercontent.com/88564663/137233037-7cb9cb58-9163-44fc-98c7-59a33bccd704.png)
<br>
-Mô hình dự đoán kết quả<br><br>
#### 2.4.3 Dữ liệu cost_revenue_clean.csv
![ccost](https://user-images.githubusercontent.com/88564663/137233121-5f5df854-a010-4856-b224-3127214f0b05.png)<br>
-Nạp dữ liệu file cost_revenue_clean.csv<br><br><br>
![vcost](https://user-images.githubusercontent.com/88564663/137233163-cf742290-2c7c-445b-8425-43c658b64684.png)<br>
-Mô hình dự đoán kết quả.
