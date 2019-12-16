
<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />

<title>Final_Project_Code</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>



<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.7.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.7.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.7.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff2?v=4.7.0') format('woff2'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.7.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.7.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.7.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.fa-pull-left {
  float: left;
}
.fa-pull-right {
  float: right;
}
.fa.fa-pull-left {
  margin-right: .3em;
}
.fa.fa-pull-right {
  margin-left: .3em;
}
/* Deprecated as of 4.4.0 */
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
.fa-pulse {
  -webkit-animation: fa-spin 1s infinite steps(8);
  animation: fa-spin 1s infinite steps(8);
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=1)";
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2)";
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=3)";
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)";
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)";
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook-f:before,
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-feed:before,
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before,
.fa-gratipay:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper-pp:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-resistance:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-y-combinator-square:before,
.fa-yc-square:before,
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
.fa-buysellads:before {
  content: "\f20d";
}
.fa-connectdevelop:before {
  content: "\f20e";
}
.fa-dashcube:before {
  content: "\f210";
}
.fa-forumbee:before {
  content: "\f211";
}
.fa-leanpub:before {
  content: "\f212";
}
.fa-sellsy:before {
  content: "\f213";
}
.fa-shirtsinbulk:before {
  content: "\f214";
}
.fa-simplybuilt:before {
  content: "\f215";
}
.fa-skyatlas:before {
  content: "\f216";
}
.fa-cart-plus:before {
  content: "\f217";
}
.fa-cart-arrow-down:before {
  content: "\f218";
}
.fa-diamond:before {
  content: "\f219";
}
.fa-ship:before {
  content: "\f21a";
}
.fa-user-secret:before {
  content: "\f21b";
}
.fa-motorcycle:before {
  content: "\f21c";
}
.fa-street-view:before {
  content: "\f21d";
}
.fa-heartbeat:before {
  content: "\f21e";
}
.fa-venus:before {
  content: "\f221";
}
.fa-mars:before {
  content: "\f222";
}
.fa-mercury:before {
  content: "\f223";
}
.fa-intersex:before,
.fa-transgender:before {
  content: "\f224";
}
.fa-transgender-alt:before {
  content: "\f225";
}
.fa-venus-double:before {
  content: "\f226";
}
.fa-mars-double:before {
  content: "\f227";
}
.fa-venus-mars:before {
  content: "\f228";
}
.fa-mars-stroke:before {
  content: "\f229";
}
.fa-mars-stroke-v:before {
  content: "\f22a";
}
.fa-mars-stroke-h:before {
  content: "\f22b";
}
.fa-neuter:before {
  content: "\f22c";
}
.fa-genderless:before {
  content: "\f22d";
}
.fa-facebook-official:before {
  content: "\f230";
}
.fa-pinterest-p:before {
  content: "\f231";
}
.fa-whatsapp:before {
  content: "\f232";
}
.fa-server:before {
  content: "\f233";
}
.fa-user-plus:before {
  content: "\f234";
}
.fa-user-times:before {
  content: "\f235";
}
.fa-hotel:before,
.fa-bed:before {
  content: "\f236";
}
.fa-viacoin:before {
  content: "\f237";
}
.fa-train:before {
  content: "\f238";
}
.fa-subway:before {
  content: "\f239";
}
.fa-medium:before {
  content: "\f23a";
}
.fa-yc:before,
.fa-y-combinator:before {
  content: "\f23b";
}
.fa-optin-monster:before {
  content: "\f23c";
}
.fa-opencart:before {
  content: "\f23d";
}
.fa-expeditedssl:before {
  content: "\f23e";
}
.fa-battery-4:before,
.fa-battery:before,
.fa-battery-full:before {
  content: "\f240";
}
.fa-battery-3:before,
.fa-battery-three-quarters:before {
  content: "\f241";
}
.fa-battery-2:before,
.fa-battery-half:before {
  content: "\f242";
}
.fa-battery-1:before,
.fa-battery-quarter:before {
  content: "\f243";
}
.fa-battery-0:before,
.fa-battery-empty:before {
  content: "\f244";
}
.fa-mouse-pointer:before {
  content: "\f245";
}
.fa-i-cursor:before {
  content: "\f246";
}
.fa-object-group:before {
  content: "\f247";
}
.fa-object-ungroup:before {
  content: "\f248";
}
.fa-sticky-note:before {
  content: "\f249";
}
.fa-sticky-note-o:before {
  content: "\f24a";
}
.fa-cc-jcb:before {
  content: "\f24b";
}
.fa-cc-diners-club:before {
  content: "\f24c";
}
.fa-clone:before {
  content: "\f24d";
}
.fa-balance-scale:before {
  content: "\f24e";
}
.fa-hourglass-o:before {
  content: "\f250";
}
.fa-hourglass-1:before,
.fa-hourglass-start:before {
  content: "\f251";
}
.fa-hourglass-2:before,
.fa-hourglass-half:before {
  content: "\f252";
}
.fa-hourglass-3:before,
.fa-hourglass-end:before {
  content: "\f253";
}
.fa-hourglass:before {
  content: "\f254";
}
.fa-hand-grab-o:before,
.fa-hand-rock-o:before {
  content: "\f255";
}
.fa-hand-stop-o:before,
.fa-hand-paper-o:before {
  content: "\f256";
}
.fa-hand-scissors-o:before {
  content: "\f257";
}
.fa-hand-lizard-o:before {
  content: "\f258";
}
.fa-hand-spock-o:before {
  content: "\f259";
}
.fa-hand-pointer-o:before {
  content: "\f25a";
}
.fa-hand-peace-o:before {
  content: "\f25b";
}
.fa-trademark:before {
  content: "\f25c";
}
.fa-registered:before {
  content: "\f25d";
}
.fa-creative-commons:before {
  content: "\f25e";
}
.fa-gg:before {
  content: "\f260";
}
.fa-gg-circle:before {
  content: "\f261";
}
.fa-tripadvisor:before {
  content: "\f262";
}
.fa-odnoklassniki:before {
  content: "\f263";
}
.fa-odnoklassniki-square:before {
  content: "\f264";
}
.fa-get-pocket:before {
  content: "\f265";
}
.fa-wikipedia-w:before {
  content: "\f266";
}
.fa-safari:before {
  content: "\f267";
}
.fa-chrome:before {
  content: "\f268";
}
.fa-firefox:before {
  content: "\f269";
}
.fa-opera:before {
  content: "\f26a";
}
.fa-internet-explorer:before {
  content: "\f26b";
}
.fa-tv:before,
.fa-television:before {
  content: "\f26c";
}
.fa-contao:before {
  content: "\f26d";
}
.fa-500px:before {
  content: "\f26e";
}
.fa-amazon:before {
  content: "\f270";
}
.fa-calendar-plus-o:before {
  content: "\f271";
}
.fa-calendar-minus-o:before {
  content: "\f272";
}
.fa-calendar-times-o:before {
  content: "\f273";
}
.fa-calendar-check-o:before {
  content: "\f274";
}
.fa-industry:before {
  content: "\f275";
}
.fa-map-pin:before {
  content: "\f276";
}
.fa-map-signs:before {
  content: "\f277";
}
.fa-map-o:before {
  content: "\f278";
}
.fa-map:before {
  content: "\f279";
}
.fa-commenting:before {
  content: "\f27a";
}
.fa-commenting-o:before {
  content: "\f27b";
}
.fa-houzz:before {
  content: "\f27c";
}
.fa-vimeo:before {
  content: "\f27d";
}
.fa-black-tie:before {
  content: "\f27e";
}
.fa-fonticons:before {
  content: "\f280";
}
.fa-reddit-alien:before {
  content: "\f281";
}
.fa-edge:before {
  content: "\f282";
}
.fa-credit-card-alt:before {
  content: "\f283";
}
.fa-codiepie:before {
  content: "\f284";
}
.fa-modx:before {
  content: "\f285";
}
.fa-fort-awesome:before {
  content: "\f286";
}
.fa-usb:before {
  content: "\f287";
}
.fa-product-hunt:before {
  content: "\f288";
}
.fa-mixcloud:before {
  content: "\f289";
}
.fa-scribd:before {
  content: "\f28a";
}
.fa-pause-circle:before {
  content: "\f28b";
}
.fa-pause-circle-o:before {
  content: "\f28c";
}
.fa-stop-circle:before {
  content: "\f28d";
}
.fa-stop-circle-o:before {
  content: "\f28e";
}
.fa-shopping-bag:before {
  content: "\f290";
}
.fa-shopping-basket:before {
  content: "\f291";
}
.fa-hashtag:before {
  content: "\f292";
}
.fa-bluetooth:before {
  content: "\f293";
}
.fa-bluetooth-b:before {
  content: "\f294";
}
.fa-percent:before {
  content: "\f295";
}
.fa-gitlab:before {
  content: "\f296";
}
.fa-wpbeginner:before {
  content: "\f297";
}
.fa-wpforms:before {
  content: "\f298";
}
.fa-envira:before {
  content: "\f299";
}
.fa-universal-access:before {
  content: "\f29a";
}
.fa-wheelchair-alt:before {
  content: "\f29b";
}
.fa-question-circle-o:before {
  content: "\f29c";
}
.fa-blind:before {
  content: "\f29d";
}
.fa-audio-description:before {
  content: "\f29e";
}
.fa-volume-control-phone:before {
  content: "\f2a0";
}
.fa-braille:before {
  content: "\f2a1";
}
.fa-assistive-listening-systems:before {
  content: "\f2a2";
}
.fa-asl-interpreting:before,
.fa-american-sign-language-interpreting:before {
  content: "\f2a3";
}
.fa-deafness:before,
.fa-hard-of-hearing:before,
.fa-deaf:before {
  content: "\f2a4";
}
.fa-glide:before {
  content: "\f2a5";
}
.fa-glide-g:before {
  content: "\f2a6";
}
.fa-signing:before,
.fa-sign-language:before {
  content: "\f2a7";
}
.fa-low-vision:before {
  content: "\f2a8";
}
.fa-viadeo:before {
  content: "\f2a9";
}
.fa-viadeo-square:before {
  content: "\f2aa";
}
.fa-snapchat:before {
  content: "\f2ab";
}
.fa-snapchat-ghost:before {
  content: "\f2ac";
}
.fa-snapchat-square:before {
  content: "\f2ad";
}
.fa-pied-piper:before {
  content: "\f2ae";
}
.fa-first-order:before {
  content: "\f2b0";
}
.fa-yoast:before {
  content: "\f2b1";
}
.fa-themeisle:before {
  content: "\f2b2";
}
.fa-google-plus-circle:before,
.fa-google-plus-official:before {
  content: "\f2b3";
}
.fa-fa:before,
.fa-font-awesome:before {
  content: "\f2b4";
}
.fa-handshake-o:before {
  content: "\f2b5";
}
.fa-envelope-open:before {
  content: "\f2b6";
}
.fa-envelope-open-o:before {
  content: "\f2b7";
}
.fa-linode:before {
  content: "\f2b8";
}
.fa-address-book:before {
  content: "\f2b9";
}
.fa-address-book-o:before {
  content: "\f2ba";
}
.fa-vcard:before,
.fa-address-card:before {
  content: "\f2bb";
}
.fa-vcard-o:before,
.fa-address-card-o:before {
  content: "\f2bc";
}
.fa-user-circle:before {
  content: "\f2bd";
}
.fa-user-circle-o:before {
  content: "\f2be";
}
.fa-user-o:before {
  content: "\f2c0";
}
.fa-id-badge:before {
  content: "\f2c1";
}
.fa-drivers-license:before,
.fa-id-card:before {
  content: "\f2c2";
}
.fa-drivers-license-o:before,
.fa-id-card-o:before {
  content: "\f2c3";
}
.fa-quora:before {
  content: "\f2c4";
}
.fa-free-code-camp:before {
  content: "\f2c5";
}
.fa-telegram:before {
  content: "\f2c6";
}
.fa-thermometer-4:before,
.fa-thermometer:before,
.fa-thermometer-full:before {
  content: "\f2c7";
}
.fa-thermometer-3:before,
.fa-thermometer-three-quarters:before {
  content: "\f2c8";
}
.fa-thermometer-2:before,
.fa-thermometer-half:before {
  content: "\f2c9";
}
.fa-thermometer-1:before,
.fa-thermometer-quarter:before {
  content: "\f2ca";
}
.fa-thermometer-0:before,
.fa-thermometer-empty:before {
  content: "\f2cb";
}
.fa-shower:before {
  content: "\f2cc";
}
.fa-bathtub:before,
.fa-s15:before,
.fa-bath:before {
  content: "\f2cd";
}
.fa-podcast:before {
  content: "\f2ce";
}
.fa-window-maximize:before {
  content: "\f2d0";
}
.fa-window-minimize:before {
  content: "\f2d1";
}
.fa-window-restore:before {
  content: "\f2d2";
}
.fa-times-rectangle:before,
.fa-window-close:before {
  content: "\f2d3";
}
.fa-times-rectangle-o:before,
.fa-window-close-o:before {
  content: "\f2d4";
}
.fa-bandcamp:before {
  content: "\f2d5";
}
.fa-grav:before {
  content: "\f2d6";
}
.fa-etsy:before {
  content: "\f2d7";
}
.fa-imdb:before {
  content: "\f2d8";
}
.fa-ravelry:before {
  content: "\f2d9";
}
.fa-eercast:before {
  content: "\f2da";
}
.fa-microchip:before {
  content: "\f2db";
}
.fa-snowflake-o:before {
  content: "\f2dc";
}
.fa-superpowers:before {
  content: "\f2dd";
}
.fa-wpexplorer:before {
  content: "\f2de";
}
.fa-meetup:before {
  content: "\f2e0";
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
div.traceback-wrapper pre.traceback {
  max-height: 600px;
  overflow: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  padding: 5px;
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
[dir="rtl"] #ipython_notebook {
  margin-right: 10px;
  margin-left: 0;
}
[dir="rtl"] #ipython_notebook.pull-left {
  float: right !important;
  float: right;
}
.flex-spacer {
  flex: 1;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#kernel_logo_widget {
  margin: 0 10px;
}
span#login_widget {
  float: right;
}
[dir="rtl"] span#login_widget {
  float: left;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
.modal-header {
  cursor: move;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
[dir="rtl"] .center-nav form.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] .center-nav .navbar-text {
  float: right;
}
[dir="rtl"] .navbar-inner {
  text-align: right;
}
[dir="rtl"] div.text-left {
  text-align: right;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  position: absolute;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  cursor: pointer;
  opacity: 0;
  z-index: 2;
}
.alternate_upload .btn-xs > input.fileinput {
  margin: -1px -5px;
}
.alternate_upload .btn-upload {
  position: relative;
  height: 22px;
}
::-webkit-file-upload-button {
  cursor: pointer;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
ul#tabs {
  margin-bottom: 4px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
[dir="rtl"] ul#tabs.nav-tabs > li {
  float: right;
}
[dir="rtl"] ul#tabs.nav.nav-tabs {
  padding-right: 0;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons .pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .list_toolbar .col-sm-4,
[dir="rtl"] .list_toolbar .col-sm-8 {
  float: right;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: text-bottom;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
[dir="rtl"] .list_item > div input {
  margin-right: 0;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_modified {
  margin-right: 7px;
  margin-left: 7px;
}
[dir="rtl"] .item_modified.pull-right {
  float: left !important;
  float: left;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
[dir="rtl"] .item_buttons.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .item_buttons .kernel-name {
  margin-left: 7px;
  float: right;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
.sort_button {
  display: inline-block;
  padding-left: 7px;
}
[dir="rtl"] .sort_button.pull-right {
  float: left !important;
  float: left;
}
#tree-selector {
  padding-right: 0px;
}
#button-select-all {
  min-width: 50px;
}
[dir="rtl"] #button-select-all.btn {
  float: right ;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
  margin-top: 2px;
  height: 16px;
}
[dir="rtl"] #select-all.pull-left {
  float: right !important;
  float: right;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.fa-pull-left {
  margin-right: .3em;
}
.folder_icon:before.fa-pull-right {
  margin-left: .3em;
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.fa-pull-left {
  margin-right: .3em;
}
.file_icon:before.fa-pull-right {
  margin-left: .3em;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
#new-menu .dropdown-header {
  font-size: 10px;
  border-bottom: 1px solid #e5e5e5;
  padding: 0 0 3px;
  margin: -3px 20px 0;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.move-button {
  display: none;
}
.download-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
.CodeMirror-dialog {
  background-color: #fff;
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI escape sequences */
/* The color values are a mix of
   http://www.xcolors.net/dl/baskerville-ivorylight and
   http://www.xcolors.net/dl/euphrasia */
.ansi-black-fg {
  color: #3E424D;
}
.ansi-black-bg {
  background-color: #3E424D;
}
.ansi-black-intense-fg {
  color: #282C36;
}
.ansi-black-intense-bg {
  background-color: #282C36;
}
.ansi-red-fg {
  color: #E75C58;
}
.ansi-red-bg {
  background-color: #E75C58;
}
.ansi-red-intense-fg {
  color: #B22B31;
}
.ansi-red-intense-bg {
  background-color: #B22B31;
}
.ansi-green-fg {
  color: #00A250;
}
.ansi-green-bg {
  background-color: #00A250;
}
.ansi-green-intense-fg {
  color: #007427;
}
.ansi-green-intense-bg {
  background-color: #007427;
}
.ansi-yellow-fg {
  color: #DDB62B;
}
.ansi-yellow-bg {
  background-color: #DDB62B;
}
.ansi-yellow-intense-fg {
  color: #B27D12;
}
.ansi-yellow-intense-bg {
  background-color: #B27D12;
}
.ansi-blue-fg {
  color: #208FFB;
}
.ansi-blue-bg {
  background-color: #208FFB;
}
.ansi-blue-intense-fg {
  color: #0065CA;
}
.ansi-blue-intense-bg {
  background-color: #0065CA;
}
.ansi-magenta-fg {
  color: #D160C4;
}
.ansi-magenta-bg {
  background-color: #D160C4;
}
.ansi-magenta-intense-fg {
  color: #A03196;
}
.ansi-magenta-intense-bg {
  background-color: #A03196;
}
.ansi-cyan-fg {
  color: #60C6C8;
}
.ansi-cyan-bg {
  background-color: #60C6C8;
}
.ansi-cyan-intense-fg {
  color: #258F8F;
}
.ansi-cyan-intense-bg {
  background-color: #258F8F;
}
.ansi-white-fg {
  color: #C5C1B4;
}
.ansi-white-bg {
  background-color: #C5C1B4;
}
.ansi-white-intense-fg {
  color: #A1A6B2;
}
.ansi-white-intense-bg {
  background-color: #A1A6B2;
}
.ansi-default-inverse-fg {
  color: #FFFFFF;
}
.ansi-default-inverse-bg {
  background-color: #000000;
}
.ansi-bold {
  font-weight: bold;
}
.ansi-underline {
  text-decoration: underline;
}
/* The following styles are deprecated an will be removed in a future version */
.ansibold {
  font-weight: bold;
}
.ansi-inverse {
  outline: 0.5px dotted;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  position: relative;
  overflow: visible;
}
div.cell:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: transparent;
}
div.cell.jupyter-soft-selected {
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected,
div.cell.selected.jupyter-soft-selected {
  border-color: #ababab;
}
div.cell.selected:before,
div.cell.selected.jupyter-soft-selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #42A5F5;
}
@media print {
  div.cell.selected,
  div.cell.selected.jupyter-soft-selected {
    border-color: transparent;
  }
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
}
.edit_mode div.cell.selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #66BB6A;
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  /* Note that this should set vertical padding only, since CodeMirror assumes
       that horizontal padding will be set on CodeMirror pre */
  padding: 0.4em 0;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. This sets horizontal padding only,
    use .CodeMirror-lines for vertical */
  padding: 0 0.4em;
  border: 0;
  border-radius: 0;
}
.CodeMirror-cursor {
  border-left: 1.4px solid black;
}
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .CodeMirror-cursor {
    border-left: 2px solid black;
  }
}
@media screen and (min-width: 4320px) {
  .CodeMirror-cursor {
    border-left: 4px solid black;
  }
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
div.output_area .mglyph > img {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 1px 0 1px 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul:not(.list-inline),
.rendered_html ol:not(.list-inline) {
  padding-left: 2em;
}
.rendered_html ul {
  list-style: disc;
}
.rendered_html ul ul {
  list-style: square;
  margin-top: 0;
}
.rendered_html ul ul ul {
  list-style: circle;
}
.rendered_html ol {
  list-style: decimal;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin-top: 0;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
  padding: 0px;
  background-color: #fff;
}
.rendered_html code {
  background-color: #eff0f1;
}
.rendered_html p code {
  padding: 1px 5px;
}
.rendered_html pre code {
  background-color: #fff;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  color: #000;
  font-size: 100%;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: none;
  border-collapse: collapse;
  border-spacing: 0;
  color: black;
  font-size: 12px;
  table-layout: fixed;
}
.rendered_html thead {
  border-bottom: 1px solid black;
  vertical-align: bottom;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  text-align: right;
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html tbody tr:nth-child(odd) {
  background: #f5f5f5;
}
.rendered_html tbody tr:hover {
  background: rgba(66, 165, 245, 0.2);
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
.rendered_html .alert {
  margin-bottom: initial;
}
.rendered_html * + .alert {
  margin-top: 1em;
}
[dir="rtl"] .rendered_html p {
  text-align: right;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.rendered .rendered_html tr,
.text_cell.rendered .rendered_html th,
.text_cell.rendered .rendered_html td {
  max-width: none;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.text_cell .dropzone .input_area {
  border: 2px dashed #bababa;
  margin: -1px;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
.jupyter-keybindings {
  padding: 1px;
  line-height: 24px;
  border-bottom: 1px solid gray;
}
.jupyter-keybindings input {
  margin: 0;
  padding: 0;
  border: none;
}
.jupyter-keybindings i {
  padding: 6px;
}
.well code {
  background-color: #ffffff;
  border-color: #ababab;
  border-width: 1px;
  border-style: solid;
  padding: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.tags_button_container {
  width: 100%;
  display: flex;
}
.tag-container {
  display: flex;
  flex-direction: row;
  flex-grow: 1;
  overflow: hidden;
  position: relative;
}
.tag-container > * {
  margin: 0 4px;
}
.remove-tag-btn {
  margin-left: 4px;
}
.tags-input {
  display: flex;
}
.cell-tag:last-child:after {
  content: "";
  position: absolute;
  right: 0;
  width: 40px;
  height: 100%;
  /* Fade to background color of cell toolbar */
  background: linear-gradient(to right, rgba(0, 0, 0, 0), #EEE);
}
.tags-input > * {
  margin-left: 4px;
}
.cell-tag,
.tags-input input,
.tags-input button {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  box-shadow: none;
  width: inherit;
  font-size: inherit;
  height: 22px;
  line-height: 22px;
  padding: 0px 4px;
  display: inline-block;
}
.cell-tag:focus,
.tags-input input:focus,
.tags-input button:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.cell-tag::-moz-placeholder,
.tags-input input::-moz-placeholder,
.tags-input button::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.cell-tag:-ms-input-placeholder,
.tags-input input:-ms-input-placeholder,
.tags-input button:-ms-input-placeholder {
  color: #999;
}
.cell-tag::-webkit-input-placeholder,
.tags-input input::-webkit-input-placeholder,
.tags-input button::-webkit-input-placeholder {
  color: #999;
}
.cell-tag::-ms-expand,
.tags-input input::-ms-expand,
.tags-input button::-ms-expand {
  border: 0;
  background-color: transparent;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
.cell-tag[readonly],
.tags-input input[readonly],
.tags-input button[readonly],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  background-color: #eeeeee;
  opacity: 1;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  cursor: not-allowed;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button {
  height: auto;
}
select.cell-tag,
select.tags-input input,
select.tags-input button {
  height: 30px;
  line-height: 30px;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button,
select[multiple].cell-tag,
select[multiple].tags-input input,
select[multiple].tags-input button {
  height: auto;
}
.cell-tag,
.tags-input button {
  padding: 0px 4px;
}
.cell-tag {
  background-color: #fff;
  white-space: nowrap;
}
.tags-input input[type=text]:focus {
  outline: none;
  box-shadow: none;
  border-color: #ccc;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
[dir="rtl"] #kernel_logo_widget {
  float: left !important;
  float: left;
}
.modal .modal-body .move-path {
  display: flex;
  flex-direction: row;
  justify-content: space;
  align-items: center;
}
.modal .modal-body .move-path .server-root {
  padding-right: 20px;
}
.modal .modal-body .move-path .path-input {
  flex: 1;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
[dir="rtl"] #menubar .navbar-toggle {
  float: right;
}
[dir="rtl"] #menubar .navbar-collapse {
  clear: right;
}
[dir="rtl"] #menubar .navbar-nav {
  float: right;
}
[dir="rtl"] #menubar .nav {
  padding-right: 0px;
}
[dir="rtl"] #menubar .navbar-nav > li {
  float: right;
}
[dir="rtl"] #menubar .navbar-right {
  float: left !important;
}
[dir="rtl"] ul.dropdown-menu {
  text-align: right;
  left: auto;
}
[dir="rtl"] ul#new-menu.dropdown-menu {
  right: auto;
  left: 0;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
[dir="rtl"] i.menu-icon.pull-right {
  float: left !important;
  float: left;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
[dir="rtl"] ul#help_menu li a {
  padding-left: 2.2em;
}
[dir="rtl"] ul#help_menu li a i {
  margin-right: 0;
  margin-left: -1.2em;
}
[dir="rtl"] ul#help_menu li a i.pull-right {
  float: left !important;
  float: left;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
[dir="rtl"] .dropdown-submenu > .dropdown-menu {
  right: 100%;
  margin-right: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.fa-pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.fa-pull-right {
  margin-left: .3em;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
[dir="rtl"] .dropdown-submenu > a:after {
  float: left;
  content: "\f0d9";
  margin-right: 0;
  margin-left: -10px;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
[dir="rtl"] #notification_area {
  float: left !important;
  float: left;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] .indicator_area {
  float: left !important;
  float: left;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
[dir="rtl"] #kernel_indicator {
  float: left !important;
  float: left;
  border-left: 0;
  border-right: 1px solid;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] #modal_indicator {
  float: left !important;
  float: left;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  height: 30px;
  margin-top: 4px;
  display: flex;
  justify-content: flex-start;
  align-items: baseline;
  width: 50%;
  flex: 1;
}
span.save_widget span.filename {
  height: 100%;
  line-height: 1em;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
[dir="rtl"] span.save_widget.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] span.save_widget span.filename {
  margin-left: 0;
  margin-right: 16px;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
  white-space: nowrap;
  padding: 0 5px;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
    padding: 0 0 0 5px;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
.toolbar-btn-label {
  margin-left: 6px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
[dir="rtl"] .btn-group > .btn,
.btn-group-vertical > .btn {
  float: right;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
[dir="rtl"] ul.typeahead-list i {
  margin-left: 0;
  margin-right: -10px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
ul.typeahead-list  > li > a.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .typeahead-list {
  text-align: right;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  min-width: 20px;
  color: transparent;
}
[dir="rtl"] .no-shortcut.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .command-shortcut.pull-right {
  float: left !important;
  float: left;
}
.command-shortcut:before {
  content: "(command mode)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
[dir="rtl"] .edit-shortcut.pull-right {
  float: left !important;
  float: left;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
[dir="ltr"] #find-and-replace .input-group-btn + .form-control {
  border-left: none;
}
[dir="rtl"] #find-and-replace .input-group-btn + .form-control {
  border-right: none;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="https://miro.medium.com/max/3840/1*e3E0OQzfYCuWk0pket5dAA.png" alt="Reddit Logo" /></p>
<p><center> <h1 style="font-size:36px;">Predicting Upvotes and Popularity on Reddit</h1> </center></p>
<p><h4>
Authors: Andrew Paul, Chigozie Nna</h4></p>
<p><hr></p>
<body>
<h1>Introduction </h1>

Reddit is an American social news aggregation, web content rating, and discussion website. Reddit originally created by two University of Virgina Students, Steven Huffman and Alexis Ohanian, in the year 2005. A year later Condé Nast Publications acquired the site as their own. Popularity in Reddit began to arise, as by 2007 NSFW, Programming, and Science where the the top trending subreddits of the time. By the year 2008, a launch of numerous different subreddits began to popularize the site, with Reddit being able to gain enough popularity to overtake Digg in search popularity by 2010. Reddit’s rise to fame did not stop there with, Reddit finally achieving a total of one billion page views per month in the year 2011. The goal of Reddit is for members to be able to submit content to the site in the form of links, text posts, and images, which can then be voted up or down by opposing members. The posts are categorized into items called “Subreddits” where users can share specific topics and/or interests that relate to the category at hand. Full details on it’s timeline and history can be viewed here.

In this tutorial, our goal is to tidy up the data of posts within a years total, to provide us with knowledge into which what the amount of characters in a post cause the most effect in terms of up votes, down votes, score, and in general a reaction to the post. Post may vary in topics, arguments, time posted, and many more varieties, but we feel as if the popularity really depends on the length of characters, time and date, and topic of the post. We will be able to determine which length is just to short, and what length is long enough to bore an audience and not give the time to react to it.We will also be able to  look at the most popular subreddit posts and time of day to see the upvote relation there. We hope to give enough information and analysis to provide, clarity, understanding and hopefully a new found interest to readers that are unfamiliar with the social foreground. Hopefully those who are frequent Reddit users, will gain some insight on how long they should make their posts if they are trying to gain more popularity.
<body>
<hr>
<body>
<h1 id="getting-started-with-the-data">Getting started with the Data</h1>
<p>We decided to use Python 3 and SQL to help gain and analyze our data. Crucial libraries used to help us where: <a href="https://towardsdatascience.com/a-quick-introduction-to-the-pandas-python-library-f1b678f34673">pandas</a>, <a href="https://matplotlib.org/">matplotlib</a>, <a href="https://python-graph-gallery.com/seaborn/">seaborn</a>, and <a href="https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/">scikit-learn</a>.</p>
</body>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[166]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="k">import</span> <span class="n">model_selection</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="k">import</span> <span class="n">linear_model</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="k">import</span> <span class="n">preprocessing</span>
<span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="k">import</span> <span class="n">LabelEncoder</span>
<span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="k">import</span> <span class="n">PolynomialFeatures</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LinearRegression</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<body> We plan on using multiple panda dataframes that will be read in using SQL commands through googles BigQuery website.

<h2>Processing and Recieving data </h2>

We used the following SQL command through Googles BigQuery to at first take data from a third party called <a href=https://pushshift.io/>"Pushshift"</a> that is a Reddit API that tracks almost all of Reddit's for the last few years. We are taking in data from 2016 to Augst 2019 due to the immense amount of data that is tracked.
</body>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src= "https://i.imgur.com/xc6mlpA.png>" alt= "SQL Code" width="400"/></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><body> In this SQL Query we are getting the length of every single title, averaging the score based on the length of the title, the average number of comments based on the length of the title, and the number of posts with that amount of characters. This is done by using the 'GROUP BY' command with SQL. BigQuery convertd this data into a <a href= https://www.howtogeek.com/348960/what-is-a-csv-file-and-how-do-i-open-it/>csv file</a>, which is a table or excel seperating the data by commas (,) making it easy to parse and split the data with.</p>
<h2> Reading the Data </h2><p>We will First use Pythons Pandas to read in the csv file and convert it into a panda <a href=https://www.geeksforgeeks.org/python-pandas-dataframe/>dataframe</a>, which is a two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns)</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[167]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;LengthScoreComments.csv&#39;</span><span class="p">,</span> <span class="n">sep</span><span class="o">=</span><span class="s1">&#39;,&#39;</span><span class="p">)</span>
<span class="n">data</span><span class="p">[:</span><span class="mi">10</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[167]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>length_title</th>
      <th>avg_score</th>
      <th>avg_comments</th>
      <th>num_posts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>40.804163</td>
      <td>1.789499</td>
      <td>472705</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>65.521526</td>
      <td>2.796440</td>
      <td>739424</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>65.577614</td>
      <td>3.039975</td>
      <td>1361269</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>92.408734</td>
      <td>3.559541</td>
      <td>2588850</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>84.223042</td>
      <td>3.203536</td>
      <td>2210213</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>129.382588</td>
      <td>3.370371</td>
      <td>3850745</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>105.844544</td>
      <td>3.985902</td>
      <td>2858907</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>99.618349</td>
      <td>4.208025</td>
      <td>2939891</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>104.936504</td>
      <td>4.531499</td>
      <td>3818682</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>103.995719</td>
      <td>4.758954</td>
      <td>3773035</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In the DataFrame above you can see:</p>
<ul>
<li>length_title: Amount of Characters in the title</li>
<li>avg_score: The average Score the post will recieve with character length</li>
<li>avg_comments: The average amount of comments a post will recieve with character length</li>
<li>num_posts: The Number of posts between 2016-Aug 2019 with character Length</li>
</ul>
<p><hr size="20"></p>
<body>
<h2> Graphing</h2>

In this first graph we will graph to see the relation between Length of Title verse the Average Score to see if there is a relation between if a reddit user will reciever more votes based on the character length of their post. This can help readers get an insight to how long they should make their posts if they desire to be the most popular reddit user of their peers.
</body>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[66]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;length_title&#39;</span><span class="p">]</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;avg_score&#39;</span><span class="p">]</span>
<span class="n">Size</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;num_posts&#39;</span><span class="p">]</span><span class="o">/</span><span class="mi">200000</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">Y</span><span class="p">,</span> <span class="n">s</span> <span class="o">=</span> <span class="n">Size</span> <span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Length of Post Title vs Average Score of Post&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Length of Post Title (# of Characters)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Average Score of Post&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><body> We can see that there is a clear relation between the upvotes and the  number of characters in the post title. Based on the graph it seems that it is the best to have captions of Character lengths between 5-25 and 153-300+. The range most likely is based off the fact that the type of posts are farely different. There are many popular subreddits named short and quick things like "meow" to follow a trend that will be a picture of a cat that revieve many likes as long as they are following the trend. This Explains the peak and the downtrend of the number of likes as the posts begin to be normal and causal day to day type of posts. The number of UpVotes however does begin to rise again as the number of character are longer. This is because as the characters get longer, they tend to be actual issues and problems that recieve more views and reactions (ex: A president trump quote = more characters and responses).</p>
<body>

<hr size="20">

<h2> Comments </h2><p> We then decided to check if there was a relation between amount of comments verse the character length of the posts. People will place upvotes to anything they think is funny, however we wanted to see which posts actually get people commenting. Comments are what we deemed as true reactions to the posts, since it requires viewers to put in more effort than just a click for an upvote.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[67]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;length_title&#39;</span><span class="p">]</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;avg_comments&#39;</span><span class="p">]</span>
<span class="n">Size</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;num_posts&#39;</span><span class="p">]</span><span class="o">/</span><span class="mi">200000</span>
<span class="n">plot</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">Y</span><span class="p">,</span> <span class="n">s</span> <span class="o">=</span> <span class="n">Size</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Avg Number of Comments vs Length of Post title&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Length of Post Title (# of Characters)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Average Comments&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<body>Based on the relation we are able to see that the Number of comments are actually greater based on the character length of the posts. A reasoning for this data is that the longer the character length in posts most likely mean, it is on a controversial topic, quote, or a scientific analysis that people are more likely to comment on and put their input. This is in contrast to small memes and jokes that will mostly recieve likes and not comments.</body><hr size=20>

<p><center> <h2> Reddit Artwork </h2> </center>
<img src="https://static.makeuseof.com/wp-content/uploads/2019/04/whats-reddit-670x335.jpg" /></p>
<h2> Filtered Data </h2><p> We then decided to filter our data because there was very big bias with the top subreddits with small captions. We want to see the relationship between length and upvotes for typical reddit day to day users, that are making posts that arent silly small trends like "meow"</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><img src="https://i.imgur.com/bALsUPt.png" width="500" /></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><p> In this SQL Query we are creating an entirely new cvs file with the top 15 subreddits to see the relation between these subreddits that people typically talk about on a day to day basis. This data should give an accurate analysis of what we were looking for and give a different result in the relation shown in the graph<p></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[170]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Unbias</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;FilteredSubLengthScore.csv&#39;</span><span class="p">,</span> <span class="n">sep</span><span class="o">=</span><span class="s1">&#39;,&#39;</span><span class="p">)</span>
<span class="n">Unbias</span><span class="p">[:</span><span class="mi">10</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[170]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>length_title</th>
      <th>avg_score</th>
      <th>num_posts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>90.522604</td>
      <td>9401</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>210.944914</td>
      <td>18934</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>207.167737</td>
      <td>35812</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>218.983280</td>
      <td>61842</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>241.415093</td>
      <td>57298</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>271.754140</td>
      <td>62556</td>
    </tr>
    <tr>
      <td>6</td>
      <td>7</td>
      <td>285.368175</td>
      <td>77455</td>
    </tr>
    <tr>
      <td>7</td>
      <td>8</td>
      <td>288.794141</td>
      <td>94686</td>
    </tr>
    <tr>
      <td>8</td>
      <td>9</td>
      <td>316.807680</td>
      <td>123414</td>
    </tr>
    <tr>
      <td>9</td>
      <td>10</td>
      <td>313.843326</td>
      <td>141357</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In the DataFrame above you can see:</p>
<ul>
<li>length_title: Amount of Characters in the title</li>
<li>avg_score: The average Score the post will recieve with character length</li>
<li>num_posts: The Number of posts between 2016-Aug 2019 with character Length</li>
</ul>
<p><hr size="20"></p>
<body>
<h2> Graphing Filtered Data</h2>

In this next graph we will graph to see the relation between Length of Title verse the Average Score to see if there is the same relation as the first graph, however if there is now a change in the data now that it is based off the top 15 most popular subreddits. This should give a more accurate amount of data that should actually help reddit users decide how loong their pots should be so they can be popular.
</body>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[70]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">Unbias</span><span class="p">[</span><span class="s1">&#39;length_title&#39;</span><span class="p">]</span>
<span class="n">Y</span> <span class="o">=</span> <span class="n">Unbias</span><span class="p">[</span><span class="s1">&#39;avg_score&#39;</span><span class="p">]</span>
<span class="n">Size</span> <span class="o">=</span> <span class="n">Unbias</span><span class="p">[</span><span class="s1">&#39;num_posts&#39;</span><span class="p">]</span><span class="o">/</span><span class="mi">10000</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">x</span> <span class="o">=</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">Y</span><span class="p">,</span> <span class="n">s</span> <span class="o">=</span> <span class="n">Size</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Length of Post Title vs Average Score of Post (Filtered)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Length of Post Title (# of Characters)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Average Score of Post&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Based on the new graph we can see a less skewed range of values and a smaller drop off between the relationship. We can see a more accurate drop off. It seems that the amount of characters required are still the same despite the data being filtered now. This overall proves the fact that it is best to have posts that are between the amounts of 5-25 characters. </p><hr size="20">

<center> <h2> Linear & Polynomial Regression </h2> </center><p> We decied that we would like to get a relative prediction of what the outcome would be by create first a linear regression of the data and then create a ploynomial trend line to the data since it is not a linear change based on the look. </p><h3> Linear </h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[71]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Y</span> <span class="o">=</span> <span class="n">Unbias</span><span class="p">[</span><span class="s1">&#39;avg_score&#39;</span><span class="p">]</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">Unbias</span><span class="p">[</span><span class="s1">&#39;length_title&#39;</span><span class="p">]</span>

<span class="n">linear_regression</span> <span class="o">=</span> <span class="n">LinearRegression</span><span class="p">()</span>
<span class="n">reshapedX</span> <span class="o">=</span> <span class="n">X</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
<span class="n">linear_regression</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">reshapedX</span><span class="p">,</span> <span class="n">Y</span><span class="p">)</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">linear_regression</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">reshapedX</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">8</span><span class="p">));</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">);</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">model</span><span class="p">);</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Length of Post Title vs Average Score of Post (Filtered)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Length of Post Title (# of Characters)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Average Score of Post&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3> Polynomial </h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[64]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">poly_reg</span> <span class="o">=</span> <span class="n">PolynomialFeatures</span><span class="p">(</span><span class="n">degree</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
<span class="n">reshapedX</span> <span class="o">=</span> <span class="n">X</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
<span class="n">poly</span> <span class="o">=</span> <span class="n">poly_reg</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">reshapedX</span><span class="p">)</span>


<span class="n">linear_regression2</span> <span class="o">=</span> <span class="n">LinearRegression</span><span class="p">()</span>
<span class="n">reshapedY</span> <span class="o">=</span> <span class="n">Y</span><span class="o">.</span><span class="n">values</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
<span class="n">linear_regression2</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">poly</span><span class="p">,</span> <span class="n">reshapedY</span><span class="p">)</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">linear_regression2</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">poly</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">8</span><span class="p">));</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">Y</span><span class="p">);</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">);</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Length of Post Title vs Average Score of Post (Filtered)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Length of Post Title (# of Characters)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Average Score of Post&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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9VJo37IEcmuGAv5ExtkFBXShBXL1BUJEpONQoJakW+etadYqVFVbz63z1uhDMY7p48qa1FF7HPPlgByDeI1wQa1YsWZRhgezsYKtoGPMPkutZiIiHY0CtSRFrjMVb7sES2UNtmyQa15bWbRgLV43Y7xgLDKYTeUYIiLScShQS5IWAE2fyICjR1E+NXX1VNY2ZLhksdU2OEqKvATuofFvyYwDSyQYa4tjiIhI9lOgliQtAJpe0QKOaInMgwbzh7aFkpXHm+EZ+bhI4WPC5iwub8yeEGlXVa1SKImISKtToJYkdTu1vlitRcnOuA3aP3ydt1CQVxalGzK0TyS1oIqISFvQOmoiMbRFxgMREenctI6aSIrUgioiIpmkQE0kDg3cFxGRTMnJdAFEREREJDoFaiIiIiJZSl2fIglQflcREckEBWoicSi/q4iIZIq6PkXiiJXfVUREpDUpUBOJQ/ldRUQkUxSoicQRlIVA2QlERKS1KVATiWPmaSMpys9tsk35XUVEpC1oMoFIHMpOICIimaJATSQByk4gIiKZoK5PERERkSylQE1EREQkSylQExEREclSCtREREREspQCNREREZEspUBNREREJEspUBMRERHJUgrURERERLKUAjURERGRLKVATURERCRLtZtAzcymmtkaM1trZrMyXR4RERGR1tYuAjUzywV+B5wOjAa+amajM1sqERERkdbVLgI14GhgrXPuI+dcDfBP4OwMl0lERESkVbWXQK0MWB/29wZ/m4iIiEiH1V4CNYuyzTXZwewyM1toZgu3bt3aRsUSERERaT15mS5AgjYAA8P+PgjYGL6Dc+4e4B4AM9tqZutasTylwLZWPH5npDpNP9Vp+qlO00v1mX6q0/RrizodHHSHOeeC7ssaZpYHvA+cApQD7wAXOudWZqg8C51zEzJx7o5KdZp+qtP0U52ml+oz/VSn6ZfpOm0XLWrOuToz+y4wD8gF/pKpIE1ERESkrbSLQA3AOfcs8GymyyEiIiLSVtrLZIJsc0+mC9ABqU7TT3WafqrT9FJ9pp/qNP0yWqftYoyaiIiISGekFjURERGRLKVALUnKOdpyZvaJmS03syVmttDf1svMXjCzD/zfPTNdzmxmZn8xsy1mtiJsW9Q6NM9d/jW7zMyOylzJs1dAnc42s3L/Wl1iZmeE3fdjv07XmNlpmSl1djOz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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><p>As you can see the ploynomial regression gives us a more accurate analysis of the data. We can see, the relation between the length of characters in a post and the average number of upvotes is a clear polynomial relation. Reddit users can use this now to get an accurate way to decide if they want to either go with a short amount of characters to get around 300 upvotes, or go with a more lengthy response to get 300+ upvotes. </p></p>
<hr size=20>

<p><center> <h2> Typical Reddit Home Page </h2></center>
<img src="https://i.redd.it/vb63xmw7skm21.png" width= 800/></p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<center> <h2> Time Matters </h2> </center><p>In this section we will be analyzing whether the time and day of the post also come in as a factor to upvotes. This way us Reddit users can pinpoint the very day and time we should post to get the maximum popularity and be seen as a legend to our peers. We first started by getting new data again through BigQuery</p><p><img src="https://i.imgur.com/63bqeZ2.png" width= 700/></p>
<p> Relationship between time of day posted and average score of the post (if score >= 100). We looked at the average score of posts on every hour of every day of the week, so 24 hours in a day x 7 days a week gives us 168 rows of data points </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[174]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">timeScore</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;TimeVsScore.csv&#39;</span><span class="p">,</span> <span class="n">sep</span><span class="o">=</span><span class="s1">&#39;,&#39;</span><span class="p">)</span>
<span class="n">formattedTimeScore</span> <span class="o">=</span> <span class="n">timeScore</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
<span class="n">formattedTimeScore</span><span class="p">[</span><span class="s1">&#39;hourofday&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">formattedTimeScore</span><span class="p">[</span><span class="s1">&#39;hourofday&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">convertHourToTime</span><span class="p">)</span>
<span class="n">formattedTimeScore</span><span class="p">[</span><span class="s1">&#39;dayofweek&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">formattedTimeScore</span><span class="p">[</span><span class="s1">&#39;dayofweek&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">convertNumToDay</span><span class="p">)</span>

<span class="c1"># Converts the numbers 0-23 to their respective times</span>
<span class="k">def</span> <span class="nf">convertHourToTime</span><span class="p">(</span><span class="n">num</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">num</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
        <span class="n">timeOfDay</span> <span class="o">=</span> <span class="s1">&#39;12 AM&#39;</span>
    <span class="k">elif</span> <span class="n">num</span> <span class="o">&lt;=</span> <span class="mi">11</span><span class="p">:</span>
        <span class="n">timeOfDay</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">num</span><span class="p">)</span> <span class="o">+</span> <span class="s1">&#39; AM&#39;</span>
    <span class="k">elif</span> <span class="n">num</span> <span class="o">==</span> <span class="mi">12</span><span class="p">:</span>
        <span class="n">timeOfDay</span> <span class="o">=</span> <span class="s1">&#39;12 PM&#39;</span>
    <span class="k">else</span><span class="p">:</span> 
        <span class="n">timeOfDay</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">num</span><span class="o">-</span><span class="mi">12</span><span class="p">)</span> <span class="o">+</span> <span class="s1">&#39; PM&#39;</span>
    <span class="k">return</span> <span class="n">timeOfDay</span>

<span class="c1"># Converts the numbers 1-7 to their respective weekdays</span>
<span class="k">def</span> <span class="nf">convertNumToDay</span><span class="p">(</span><span class="n">num</span><span class="p">):</span>
    <span class="n">weekdays</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Sunday&#39;</span><span class="p">,</span> <span class="s1">&#39;Monday&#39;</span><span class="p">,</span> <span class="s1">&#39;Tuesday&#39;</span><span class="p">,</span> <span class="s1">&#39;Wednesday&#39;</span><span class="p">,</span> <span class="s1">&#39;Thursday&#39;</span><span class="p">,</span> <span class="s1">&#39;Friday&#39;</span><span class="p">,</span> <span class="s1">&#39;Saturday&#39;</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">weekdays</span><span class="p">[</span><span class="n">num</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>

<span class="n">formattedTimeScore</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[174]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_score</th>
      <th>dayofweek</th>
      <th>hourofday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>817.979886</td>
      <td>Sunday</td>
      <td>12 AM</td>
    </tr>
    <tr>
      <td>1</td>
      <td>832.602126</td>
      <td>Sunday</td>
      <td>1 AM</td>
    </tr>
    <tr>
      <td>2</td>
      <td>926.316992</td>
      <td>Sunday</td>
      <td>2 AM</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1013.444329</td>
      <td>Sunday</td>
      <td>3 AM</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1134.617892</td>
      <td>Sunday</td>
      <td>4 AM</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>163</td>
      <td>971.682509</td>
      <td>Saturday</td>
      <td>7 PM</td>
    </tr>
    <tr>
      <td>164</td>
      <td>942.749931</td>
      <td>Saturday</td>
      <td>8 PM</td>
    </tr>
    <tr>
      <td>165</td>
      <td>911.208568</td>
      <td>Saturday</td>
      <td>9 PM</td>
    </tr>
    <tr>
      <td>166</td>
      <td>880.878130</td>
      <td>Saturday</td>
      <td>10 PM</td>
    </tr>
    <tr>
      <td>167</td>
      <td>823.783345</td>
      <td>Saturday</td>
      <td>11 PM</td>
    </tr>
  </tbody>
</table>
<p>168 rows × 3 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[175]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">timeScoreMatrix</span> <span class="o">=</span> <span class="n">timeScore</span><span class="o">.</span><span class="n">pivot</span><span class="p">(</span><span class="n">index</span><span class="o">=</span><span class="s1">&#39;dayofweek&#39;</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="s1">&#39;hourofday&#39;</span><span class="p">,</span> <span class="n">values</span><span class="o">=</span><span class="s1">&#39;avg_score&#39;</span><span class="p">)</span>
<span class="n">cols</span> <span class="o">=</span> <span class="n">timeScoreMatrix</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
<span class="n">cols</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">cols</span><span class="o">.</span><span class="n">pop</span><span class="p">(</span><span class="n">cols</span><span class="o">.</span><span class="n">index</span><span class="p">(</span><span class="mi">23</span><span class="p">)))</span>
<span class="n">fixedTimeScoreMatrix</span> <span class="o">=</span> <span class="n">timeScoreMatrix</span><span class="o">.</span><span class="n">reindex</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">cols</span><span class="p">)</span>
<span class="n">timeScoreMatrix</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[175]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>hourofday</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
    <tr>
      <th>dayofweek</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>817.979886</td>
      <td>832.602126</td>
      <td>926.316992</td>
      <td>1013.444329</td>
      <td>1134.617892</td>
      <td>1283.984606</td>
      <td>1400.718513</td>
      <td>1384.293555</td>
      <td>1339.667450</td>
      <td>1227.160091</td>
      <td>...</td>
      <td>940.857548</td>
      <td>968.624444</td>
      <td>937.781498</td>
      <td>929.255882</td>
      <td>967.731765</td>
      <td>942.091881</td>
      <td>964.019933</td>
      <td>963.388646</td>
      <td>853.194138</td>
      <td>802.564590</td>
    </tr>
    <tr>
      <td>2</td>
      <td>815.127267</td>
      <td>862.858846</td>
      <td>928.686203</td>
      <td>1042.522746</td>
      <td>1189.814349</td>
      <td>1386.860338</td>
      <td>1369.992439</td>
      <td>1293.379859</td>
      <td>1220.160733</td>
      <td>1125.224842</td>
      <td>...</td>
      <td>957.458976</td>
      <td>938.846684</td>
      <td>938.656955</td>
      <td>939.169165</td>
      <td>937.429680</td>
      <td>957.803950</td>
      <td>937.148762</td>
      <td>886.015139</td>
      <td>810.725206</td>
      <td>800.888918</td>
    </tr>
    <tr>
      <td>3</td>
      <td>807.148566</td>
      <td>859.623517</td>
      <td>974.795660</td>
      <td>1067.599219</td>
      <td>1179.800492</td>
      <td>1357.187672</td>
      <td>1392.787999</td>
      <td>1288.674100</td>
      <td>1203.942667</td>
      <td>1137.456890</td>
      <td>...</td>
      <td>968.803341</td>
      <td>962.026646</td>
      <td>934.731859</td>
      <td>944.079777</td>
      <td>949.844017</td>
      <td>973.267883</td>
      <td>955.275513</td>
      <td>890.547645</td>
      <td>838.271816</td>
      <td>822.564901</td>
    </tr>
    <tr>
      <td>4</td>
      <td>820.053926</td>
      <td>864.788059</td>
      <td>966.607819</td>
      <td>1086.004091</td>
      <td>1201.194851</td>
      <td>1373.418265</td>
      <td>1386.699401</td>
      <td>1285.982900</td>
      <td>1201.614869</td>
      <td>1116.236775</td>
      <td>...</td>
      <td>969.912780</td>
      <td>957.093768</td>
      <td>941.838205</td>
      <td>947.765502</td>
      <td>929.672191</td>
      <td>966.943421</td>
      <td>925.359047</td>
      <td>874.487371</td>
      <td>815.822274</td>
      <td>818.863226</td>
    </tr>
    <tr>
      <td>5</td>
      <td>810.921800</td>
      <td>883.710208</td>
      <td>969.234269</td>
      <td>1078.066438</td>
      <td>1207.298409</td>
      <td>1366.675789</td>
      <td>1366.740784</td>
      <td>1282.025036</td>
      <td>1212.700262</td>
      <td>1112.079900</td>
      <td>...</td>
      <td>958.369804</td>
      <td>932.961184</td>
      <td>935.878545</td>
      <td>949.448003</td>
      <td>930.227652</td>
      <td>964.164243</td>
      <td>938.896549</td>
      <td>885.560502</td>
      <td>840.863545</td>
      <td>813.365210</td>
    </tr>
    <tr>
      <td>6</td>
      <td>824.531597</td>
      <td>883.380917</td>
      <td>946.547062</td>
      <td>1009.453494</td>
      <td>1188.025956</td>
      <td>1347.901959</td>
      <td>1383.533918</td>
      <td>1283.798372</td>
      <td>1211.544344</td>
      <td>1119.912744</td>
      <td>...</td>
      <td>943.259381</td>
      <td>940.068199</td>
      <td>915.703810</td>
      <td>920.797151</td>
      <td>924.610139</td>
      <td>937.115934</td>
      <td>920.370725</td>
      <td>894.035268</td>
      <td>857.822298</td>
      <td>790.034420</td>
    </tr>
    <tr>
      <td>7</td>
      <td>799.677349</td>
      <td>843.552096</td>
      <td>933.552917</td>
      <td>1037.488966</td>
      <td>1152.136520</td>
      <td>1315.464118</td>
      <td>1392.310930</td>
      <td>1390.827688</td>
      <td>1304.164204</td>
      <td>1162.523854</td>
      <td>...</td>
      <td>947.246423</td>
      <td>940.004676</td>
      <td>945.921120</td>
      <td>943.630010</td>
      <td>936.120690</td>
      <td>971.682509</td>
      <td>942.749931</td>
      <td>911.208568</td>
      <td>880.878130</td>
      <td>823.783345</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 24 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[176]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">matplotlib.ticker</span> <span class="k">as</span> <span class="nn">ticker</span>
<span class="kn">import</span> <span class="nn">matplotlib.cm</span> <span class="k">as</span> <span class="nn">cm</span>
<span class="kn">import</span> <span class="nn">matplotlib</span> <span class="k">as</span> <span class="nn">mpl</span>

<span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">15</span><span class="p">))</span>
<span class="n">heatmap</span> <span class="o">=</span> <span class="n">ax</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">fixedTimeScoreMatrix</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s1">&#39;BuPu&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xticklabels</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="s1">&#39;&#39;</span><span class="p">,</span> <span class="n">formattedTimeScore</span><span class="o">.</span><span class="n">hourofday</span><span class="o">.</span><span class="n">unique</span><span class="p">()))</span> <span class="c1"># columns</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_yticklabels</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="s1">&#39;&#39;</span><span class="p">,</span> <span class="n">formattedTimeScore</span><span class="o">.</span><span class="n">dayofweek</span><span class="o">.</span><span class="n">unique</span><span class="p">()))</span> <span class="c1"># index</span>

<span class="n">tick_spacing</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">ax</span><span class="o">.</span><span class="n">xaxis</span><span class="o">.</span><span class="n">set_major_locator</span><span class="p">(</span><span class="n">ticker</span><span class="o">.</span><span class="n">MultipleLocator</span><span class="p">(</span><span class="n">tick_spacing</span><span class="p">))</span>
<span class="n">ax</span><span class="o">.</span><span class="n">yaxis</span><span class="o">.</span><span class="n">set_major_locator</span><span class="p">(</span><span class="n">ticker</span><span class="o">.</span><span class="n">MultipleLocator</span><span class="p">(</span><span class="n">tick_spacing</span><span class="p">))</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s2">&quot;Time of Day Posted vs Average Score&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Time of Day Posted (EST)&#39;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s1">&#39;Day of Week Posted&#39;</span><span class="p">)</span>

<span class="kn">from</span> <span class="nn">mpl_toolkits.axes_grid1</span> <span class="k">import</span> <span class="n">make_axes_locatable</span>
<span class="n">divider</span> <span class="o">=</span> <span class="n">make_axes_locatable</span><span class="p">(</span><span class="n">ax</span><span class="p">)</span>
<span class="n">cax</span> <span class="o">=</span> <span class="n">divider</span><span class="o">.</span><span class="n">append_axes</span><span class="p">(</span><span class="s2">&quot;right&quot;</span><span class="p">,</span> <span class="s2">&quot;3%&quot;</span><span class="p">,</span> <span class="n">pad</span><span class="o">=</span><span class="s2">&quot;1%&quot;</span><span class="p">)</span>
<span class="n">fig</span><span class="o">.</span><span class="n">colorbar</span><span class="p">(</span><span class="n">heatmap</span><span class="p">,</span> <span class="n">cax</span><span class="o">=</span><span class="n">cax</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_text output_subarea ">
<pre>&lt;Figure size 432x288 with 0 Axes&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As we can see from the data, the most popular times to post are between 6am-9am. 7am definitely has the highest ratio of upvotes over the seven days of the week, but it seems to be the most popular to make posts on Saturday and Sunday. After heavy analysis over all of these datasets we are finally able to make an accurate guess/prediction on what is best for Reddit users popularity in posts. Overall Reddit users should have the best ratios of upvotes if they post on a Saturday, at 7am, with roughly 5-25 characters in their post. This will give them the highest chance and possibility of getting a great amount of upvotes on their post, especially if it is a part of the top fifteen subreddits.

<hr size=20>

<center> <h2> Conclusion and More </h2> </center>

<p>Reddit has grown to be an outstanding social media website, however many still do not know the trick to nailing down how to get their posts viewed and seen by the public. This project helped us learn so much more about the website and what kind of data is continously being drawn by third party websites. As you can see from this analysis, there are many factors that can put into a post on reddit that will ultimately decide how many views and upvotes one will get. We hope that this helps new Reddit users get a great jump on how to begin posting and what kind of times and amount of characters they should be using based on the topic.

If you are interested in Reddit and the many datasets to use, we recommend using Google's <a href =https://cloud.google.com/bigquery/docs/> BigQuery </a> and <a href=https://pushshift.io/>Pushshift</a>. This API consistently takes in data second by second, so we used only a select amount of data from 2016 to August of this year (2019). This tutorial was a minor fraction of the amount of data and things that can be done using Reddits data. We hope that others are inspired to do the same kind of tutorials, and we hope it was worth the read! </p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span> 
</pre></div>

    </div>
</div>
</div>

</div>
    </div>
  </div>
</body>

 


</html>
