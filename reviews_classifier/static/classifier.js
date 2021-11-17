$(document).ready(function () {
    // Send insert/update ajax query
    $('#review_form').submit(function (event) {
        $('#loading').attr("hidden", false);
        $('#send_text').attr("disabled", true);
        $('#score').attr("hidden", true);
        event.preventDefault()
        let form = $("#review_form").serialize();


        console.log($("#review_text").val())
        // Insert query
        $.ajax({
            url: "/predict",
            dataType: "json",
            data: form,
            type: "POST",
            success: function (response) {
                console.log(response)
                $('#score').html('Predicted score:' + response['score'])
                $('#detected_language').html('Detected language: <b>' + response['language'] + '</b>')

                $('#detected_language').attr("hidden", false);
                $('#loading').attr("hidden", true);
                $('#score').attr("hidden", false);
                $('#send_text').attr("disabled", false);
            },
            error: function (response) {
                $('#detected_language').attr("hidden", true);
                $('#loading').attr("hidden", true);
                $('#score').html('Error in last prediction')
                $('#send_text').attr("disabled", false);
                alert(response.responseText)
            }
        })
    });
});