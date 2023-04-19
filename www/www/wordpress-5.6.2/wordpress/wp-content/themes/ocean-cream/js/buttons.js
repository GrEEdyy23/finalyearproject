// Select element with Id hamburger-cream and run mobinu function when someone click on this element (in this case span).
// And mobinu function say 'if menu is hiden and someone click, open it but if is open, hide it'
// https://www.w3schools.com/jsref/event_onclick.asp

document.getElementById('ocean_cream_mobile_menu_icon').addEventListener('click', mobinu);

function mobinu() {
  var x = document.getElementById('ocean_cream_mobile_menu');
  if (x.style.display === 'block') {
    x.style.display = 'none';
  } else {
    x.style.display = 'block';
  }
}

document.getElementById('ocean_cream_sidebar_open').addEventListener('click', sidiBer);

function sidiBer() {
  var x = document.getElementById('ocean_cream_sidebar');
  if (x.style.display === 'block') {
    x.style.display = 'none';
  } else {
    x.style.display = 'block';
  }
}
