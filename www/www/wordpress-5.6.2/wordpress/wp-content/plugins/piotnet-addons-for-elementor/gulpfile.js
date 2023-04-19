// Include gulp
var gulp = require('gulp');

// Include Our Plugins
var sass = require('gulp-sass');
var concat = require('gulp-concat');
var rename = require('gulp-rename');
var livereload = require('gulp-livereload');
var minifyCss = require('gulp-cssnano');
var uglify = require('gulp-uglify');
var autoprefixer = require('gulp-autoprefixer');
var sourcemaps = require('gulp-sourcemaps');

gulp.task('compress', function (cb) {
  return gulp.src(['assets/js/*.js'])
        .pipe(uglify())
        .pipe(rename({ suffix: '.min' }))
        .pipe(gulp.dest('assets/js/minify/'))
        .pipe(livereload());
});

// Compile Our Sass
gulp.task('sass', function () {
    return gulp.src('assets/css/*.css')
        .pipe(sass().on( "error", sass.logError ))
        .pipe(autoprefixer({
            browsers: [
                'last 3 versions',
                'iOS >= 8',
                'Safari >= 8',
                'ie >= 11',
            ]
        }))
        .pipe(minifyCss({zindex: false}))
        .pipe(rename({ suffix: '.min' }))
        .pipe(gulp.dest('assets/css/minify/'))
        .pipe(livereload());
});

// Watch Files For Changes
gulp.task('watch', function () {
    livereload.listen();
    gulp.watch(['assets/css/*.css'], ['sass']);
    gulp.watch(['assets/js/*.js'], ['compress']);
});

// Default Task
gulp.task('default', ['watch']);