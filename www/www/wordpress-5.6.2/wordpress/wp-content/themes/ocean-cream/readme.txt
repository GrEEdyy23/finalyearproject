=== Ocean Cream ===
Contributors: onliniak
Donate link: https://onliniak.github.io/donations
Tags: featured-images, left-sidebar, custom-background, translation-ready, blog
Requires at least: 4.7
Tested up to: 5.1.1
Requires PHP: 5.6
Stable tag: 1.0.6
License: GPLv3 or later
License URI: <http://www.gnu.org/licenses/gpl-3.0.html>

== Description ==
Never waste space on your blog again! Choose hidden comments and sidebar, choose Ocean Cream. In addition, thanks to its lightness, it will speed up your website.

== Frequently Asked Questions ==

= Supported browsers =

TL;DR all CSS3 browsers.

Firefox 6+, Opera "Classic" 11.64-12.18 (ugly fonts), Chrome 30+/Vivaldi/"New" Opera, IE 9+, WebKit( Midori 0.5, Luakit 1.8.0, Arora 0.11, Safari 9.1.3+ etc.), tested on latest Firefox (Linux).

= Tested screen resolutions =

1366x768 (mainly, laptop), 1920x1080, 1440x900, 600x831 (Samsung Galaxy Tab 4 7.0, mainly), Motorola Moto G (falcon).

= Problems =

- Sticky post display only "Sticky" note.
- Author info without special style, low priority. You can use .ocean_cream_info CSS class. 
- I don't recommend screen readers and text-browsers like lynks. It's non-accessibility theme.
- Max 4 columns in "classic" editor.
- No support for custom post formats. 
- Single/Inline quotes broken in mobile, when links are too long. 

= Compatibility with CSS 2.1/HTML4 browsers =

It works (reading posts, correct colors) … but 
https://imgur.com/drqEqBK.png → post
Images size, headers size, sticky button, mobile menu/sidebar button, comments don't work.

= Help, my plugins don't works (load in footer) =

Some plugins use get_footer to load scripts in footer.
But in this case we don't use footer.php, so you must change get_footer to wp_footer .

= I want to change XYZ =

For colors, look at https://github.com/onliniak/ocean-cream/blob/github-flavoured/inc/customizer_output.php#L35 for all needed CSS selectors. Or try Customizer, by copy /inc directory and add this two lines <https://github.com/onliniak/ocean-cream/blob/github-flavoured/functions.php#L2>. Warning - insecure. 
For other changes use theme editor and favourite search engine (for example Google)
or write new support ticket.

= Why the_author_meta ? (author page) =

In the future it will be easier to change presented informations.

= I want to display Facebook/Skype/xyz in author's page. =

Example: <https://easywebdesigntutorials.com/create-your-own-author-bio-box-in-wordpress-without-a-plugin/> <http://microformats.org/wiki/hcard-examples#Skype>

Add this code to your functions.php

function ocean_cream_author_box( $user_contact ) {
// add new field to profile.php
// sanitize remove invalid charackters on input, escape in output.
 $user_contact['facebook'] = sanitize_text_field(__( 'Facebook name, example: https://www.facebook.com/name' ));
 $user_contact['skype'] = sanitize_text_field(__( 'Skype username' ));
 return $user_contact;
}
add_filter( 'user_contactmethods', 'ocean_cream_author_box' );

and this code to author.php

		<dt> <?php esc_html_e('Facebook', 'ocean-cream'); ?>   </dt>
<dd> <a class="url" href="https://www.facebook.com/<?php esc_html(the_author_meta('facebook')) ?>" rel="me">My Facebook</a> </dd>
	 ----------------
		<dt> <?php esc_html_e('Skype', 'ocean-cream'); ?>   </dt>
<dd> <a class="url" href="skype:<?php esc_html(the_author_meta('skype')) ?>?chat" rel="me">Chat with me on Skype</a> </dd>	 

= How exactly works "automatic screen" ? =

The size of the font and entry / sidebar is automatically determined based on the width of the window.

 = I have under 100 points in Google Page Insight =

 You must add Expire Headers and Gzip support to your .htaccess . Also plugins may decrease score.

 = Wrong gramatic =

 Create new ticket or even better send new pull request on Github.

== Changelog ==

= 0.9 =  Upload to WordPress repository. Again, again, again …
= 0.94 = Ready for CSP. (rewrite js buttons from onload() to span).
= 0.95 = Rewrire author page (the_author function, instead of $curauth), new pagination CSS, smaller font, hsla colors (better VA/IPS support), hide comments by default, minify and merge keyboard.js + buttons.js to bundle.min.js
= 0.96 = remove no-GPL code, WP customizer support.
= 0.96.1 = remove unverified code (WP Customizer, hsla colors and custom background color support).
= 0.96.2 = new CSS selectors → for example sidiBer is now ocean_cream_sidebar, hamburger_cream is now ocean_cream_mobile_menu_icon etc, fix transparent background after upgrade to 0.96.1
= 0.97 = Fix white letters on white background (text widgets), styles/scripts variables, new submenus, add Arial font as fallback, hide www/biography when empty (author's page), change div.menu to nav.menu, "tutorial" how add social media links, wrong urls in readme.
= 0.97.1 = Add hCard microformats to author's page (readable by browsers). Remove Railway from supported fonts and background colors from tables. Due to GPDR we cannot display e-mail, but we can display name, surname, nickname, biography, avatar and homepage url. Tested with <https://webmaster.yandex.ru/tools/microtest/> and <https://search.google.com/structured-data/testing-tool>.
= 1.0 = New minimal PHP 5.6 <https://wordpress.org/news/2019/04/minimum-php-version-update/>, restore 0.97.1.
= 1.0.1 = Tested with WordPress ß and new themeunittestdata (03.2019). 95% compatible with Gutenberg. Fix: Show excerpt, when search. Fix: Too small line-high. Fix: "Show comments" button is visible, when comments are disabled. New table border.
= 1.0.2 = FIX: Comments don't display in pages, default title "Untitled" when title is empty, HTML Validator: add ul before li in sidebar, "Some texts from widgets are overflowing the sidebar.", change table/calendar border to 2 px dotted and .rinav to .ocean_cream_next_page, small letters in Gutenberg code-block; Remove: Unecessary svg class and .OCnextpage div; CSS: Optimize, new "dark terminal mode" (based on highlight.js), remove "-style blockquotes and .screen-reader-text, re-organize CSS with CSS Tidy, create style.min.css; NEW screenshot.
= 1.0.6 = [Mobile] FIX: Black "hamburger icon", unvisible long links (temporary fix), movable page, broken inline quotes (temporary fix). [CSS] NEW: Sticky icon, animated comments form style + "post comment" button. FIX: Too small headers DELETE: info section style. [Experimental] full-page images. [SEMANTIC] Add ARIA role main to article, change section to div. [PHP] Author.php: default title "Untitled" when title is empty. Functions.php: Escape all functions.

== Inspirations ==

-   Yarn → colors, mobile version (<https://wordpress.org/themes/yarn/>)
-   "Material" colors and buttons (<https://materializecss.com/buttons.html>)
-   Twenty Nineteen → blockquotes (<https://pl.wordpress.org/themes/twentynineteen/>)
-   Google Material Design → textarea and input hover

== Resources ==

-   "Hamburger" icon made by danklammer from <https://github.com/danklammer/bytesize-icons>. Licensed under MIT.
