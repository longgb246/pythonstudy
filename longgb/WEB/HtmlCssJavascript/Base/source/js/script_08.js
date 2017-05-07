
// 1、数据定义（实际生产环境中，应由后台给出）
var data = [
	{img:'peoples_01', h2:'Creative', h3:'DUET'},
	{img:'peoples_02', h2:'Friendly', h3:'DEVIL'},
	{img:'peoples_03', h2:'Tranquilent', h3:'COMPATRIOT'},
	{img:'peoples_04', h2:'Insecure', h3:'HUSSLER'},
	{img:'peoples_05', h2:'Loving', h3:'REBEL'},
	{img:'peoples_06', h2:'Passionate', h3:'SEEKER'},
	{img:'peoples_07', h2:'Crazy', h3:'FRIEND'}
];

// 2、通用函数
var g = function(id){
	if (id.substr(0,1) == '.'){
		return document.getElementsByClassName(id.substr(1));
	}
	return document.getElementById(id);
}

// 3、添加幻灯片（所有幻灯片、按钮）
function addSliders(){
	// 3.1 获取模板
	var tpl_main = g('template_main').innerHTML.replace(/^\s*/,'').replace(/\s*$/,'');   /* 消除空白符？ */
	var tpl_ctrl = g('template_ctrl').innerHTML.replace(/^\s*/,'').replace(/\s*$/,'');

	// 3.2 定义最终输出 HTML 的变量
	var out_main = []
	var out_ctrl = []

	// 3.3 遍历所有数据，构建最终输出的 HTML
	for (var i=0; i<data.length; i++){
		var _html_main = tpl_main.replace(/{{index}}/g, i).replace(/{{pic_index}}/g, data[i].img).replace(/{{h2}}/g, data[i].h2).replace(/{{h3}}/g, data[i].h3);
		var _html_ctrl = tpl_ctrl.replace(/{{index}}/g, i).replace(/{{pic_index}}/g, data[i].img);
		out_main.push(_html_main);				/* 相当于append */
		out_ctrl.push(_html_ctrl);
	}

	// 3.4 把 HTML 回写到对应的 DOM 里面
	g('template_main').innerHTML = out_main.join('');
	g('template_ctrl').innerHTML = out_ctrl.join('');

}


// 5、幻灯片切换
function switchSlider(n){
	var main = g('main_'+n);
	var ctrl = g('ctrl_'+n);
	// 清除
	var clear_main = g('.main-i');
	var clear_ctrl = g('.ctrl-i');
	for (var i=0; i<clear_main.length; i++){
		clear_main[i].className = clear_main[i].className.replace(' main-i_active', '');
		clear_ctrl[i].className = clear_ctrl[i].className.replace(' ctrl-i_active', '');
	}
	// 附加
	main.className += ' main-i_active';
	ctrl.className += ' ctrl-i_active';
}


function movePictures(){
	var pictures = g('.pictures');
	for (var i=0; i<pictures.length; i++){
		pictures[i].style.marginTop = (-1 * pictures[i].clientHeight/2 + 'px');
	}
}

// 4、定义何时输出幻灯片
window.onload = function(){
	addSliders();
	setTimeout(function (){
		switchSlider(0);
	}, 500);
}

