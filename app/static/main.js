$(document).ready(function () {
    // initializing modal.
    $('.modal').modal();

    $('#textarea').on('keyup keypress', function (key) {
        if (key.keyCode === 13) {
            // disabling newline insertion when pressing the enter key.
            key.preventDefault();

            // searching when pressing the enter key.
            var inputData = $('#textarea').val();

            $.ajax({
                url: '/search',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(inputData),
                success: function(data) {
                    $('#code1').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[0] + '"><b>GITHUB LINK</b></a><br>' + data[1]);
                    $('#code2').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[2] + '"><b>GITHUB LINK</b></a><br>' + data[3]);
                    $('#code3').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[4] + '"><b>GITHUB LINK</b></a><br>' + data[5]);
                    $('#code4').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[6] + '"><b>GITHUB LINK</b></a><br>' + data[7]);
                    $('#code5').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[8] + '"><b>GITHUB LINK</b></a><br>' + data[9]);
                    $('#code6').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[10] + '"><b>GITHUB LINK</b></a><br>' + data[11]);
                    $('#code7').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[12] + '"><b>GITHUB LINK</b></a><br>' + data[13]);
                    $('#code8').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[14] + '"><b>GITHUB LINK</b></a><br>' + data[15]);
                    $('#code9').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[16] + '"><b>GITHUB LINK</b></a><br>' + data[17]);
                    $('#code10').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[18] + '"><b>GITHUB LINK</b></a><br>' + data[19]);
                },
                timeout: 3000
            });
        }

        // toggling button if there is text on the textarea.
        if ($('#textarea').val().trim() != '') {
            $('#search-button a').removeClass('disabled');

            // searching when pressing the enter key.
            if (key.keyCode === 13) {
                $('li').removeClass('active');
                $('#one').addClass('active');
                $('.modal').modal('open');
                $('.code').hide();
                $('#code1').show();
            }
        } else {
            $('#search-button a').addClass('disabled');
        }
    });

    // searching when pressing the search button.
    $('#search-button').on('click', function () {
        $('li').removeClass('active');
        $('#one').addClass('active');
        $('.code').hide();
        $('#code1').show();

        var inputData = $('#textarea').val();

        $.ajax({
            url: '/search',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputData),
            success: function(data) {
                $('#code1').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[0] + '"><b>GITHUB LINK</b></a><br>' + data[1]);
                $('#code2').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[2] + '"><b>GITHUB LINK</b></a><br>' + data[3]);
                $('#code3').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[4] + '"><b>GITHUB LINK</b></a><br>' + data[5]);
                $('#code4').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[6] + '"><b>GITHUB LINK</b></a><br>' + data[7]);
                $('#code5').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[8] + '"><b>GITHUB LINK</b></a><br>' + data[9]);
                $('#code6').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[10] + '"><b>GITHUB LINK</b></a><br>' + data[11]);
                $('#code7').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[12] + '"><b>GITHUB LINK</b></a><br>' + data[13]);
                $('#code8').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[14] + '"><b>GITHUB LINK</b></a><br>' + data[15]);
                $('#code9').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[16] + '"><b>GITHUB LINK</b></a><br>' + data[17]);
                $('#code10').html('<a target="_blank" rel="noopener noreferrer" href="'+ data[18] + '"><b>GITHUB LINK</b></a><br>' + data[19]);
            },
        });
    });

    // giving functionality to pagination buttons in the modal through buttons.
    var pages = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    $('li').on('click', function () {
        // converting class to 'active' when pressing numerical buttons in the modal.
        if (($(this).attr('id') !== 'previous') && ($(this).attr('id') !== 'next')) {
            $('li').removeClass('active');
            $(this).addClass('active');

            $('.code').hide();
            $('#code' + $(this).children().attr('target')).show();
        }

        // giving functionality to 'previous' button.
        if ($(this).attr('id') === 'previous') {
            for (i = 0; i < pages.length; i++) {
                if ($('li').filter(".active").attr('id') === pages[i]) {
                    $('#' + pages[i]).removeClass('active');
                    if (i == 0) {
                        $('#' + pages[9]).addClass('active');
                        $('.code').hide();
                        $('#code10').show();
                        break
                    } else {
                        $('#' + pages[i - 1]).addClass('active');
                        $('.code').hide();
                        $('#code' + (i)).show();
                        break
                    }
                }
            }
        }

        // giving functionality to 'next' button.
        if ($(this).attr('id') === 'next') {
            for (i = 0; i < pages.length; i++) {
                if ($('li').filter(".active").attr('id') === pages[i]) {
                    $('#' + pages[i]).removeClass('active');
                    if (i == 9) {
                        $('#' + pages[0]).addClass('active');
                        $('.code').hide();
                        $('#code1').show();
                        break
                    } else {
                        $('#' + pages[i + 1]).addClass('active');
                        $('.code').hide();
                        $('#code' + (i + 2)).show();
                        break
                    }
                }
            }
        }
    });

    // giving functionality to pagination buttons in the modal through keys.
    $(document).on('keydown', function (key) {
        
        // giving functionality to 'previous' button.
        if (key.keyCode === 37) {
            $('#previous').click();
        }

        // giving functionality to 'next' button.
        if (key.keyCode === 39) {
            $('#next').click();
        }
    });

});