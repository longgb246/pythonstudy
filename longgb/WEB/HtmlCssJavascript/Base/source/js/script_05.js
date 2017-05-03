 //第一步把之前的数据写成一个数组的形式,定义变量为 infos


 var infos = new Array(10);
 
  //第一次筛选，找出都是大一的信息
 infos[0] = ['小A','女',21,'大一'];
 infos[1] = ['小B','男',23,'大三'];
 infos[2] = ['小C','男',24,'大四'];
 infos[3] = ['小D','女',21,'大一'];
 infos[4] = ['小E','女',22,'大四'];
 infos[5] = ['小F','男',21,'大一'];
 infos[6] = ['小G','女',22,'大二'];
 infos[7] = ['小H','女',20,'大三'];
 infos[8] = ['小I','女',20,'大一'];
 infos[9] = ['小J','男',20,'大三'];
  

 //第二次筛选，找出都是女生的信息
 for (var i=0; i<=9; i++){
    if (infos[i][3] == '大一') {
        document.write(infos[i][0] + '</br>');
    }
 }


// var aa = setInterval(clock, 100);
// function clock(){
// 	var time=new Date();         
// 	time_local = time.toLocaleTimeString();
// 	document.getElementById("clock").value = time_local;
// }


var stop_time;
function clock(){
  var time=new Date();               	  
  document.getElementById("clock").value = time;
}
var stop_time = setInterval(clock, 100); 



var num=0;
var stop_count;
function startCount() {
	document.getElementById('count').value = num;
	num=num+1;
	stop_count = setTimeout('startCount()', 1000); 
}


 
//获取显示秒数的元素，通过定时器来更改秒数。
var back_seconds = 5;
var stop_time = setInterval(back_second, 1000);

function back_second(){
    if (back_seconds > 0){
        back_seconds--;
        document.getElementById('back_second').innerHTML = back_seconds + '秒后回到主页';
    }
    else{
		clearInterval(stop_time);		
		// open('http://hao123.com', '_self');
    }
}










