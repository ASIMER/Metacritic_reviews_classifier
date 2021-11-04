$(document).ready(function () {
    $.ajax({
        url: "/departments_crud",
        dataType: "json",
        data: {"department_id": department_id},
        type: "GET",
        success: function (response) {
            let department = response["data"][0]
            window.department_name = department["0"]
            console.log(department)
            $("#form_title").text('Create a copy of the department ' + department["0"])
            $("#department_name").val(department["0"])
        },
        error: function (response) {
            alert(response.responseText)
            $("#all_content").html("Wrong department id. Open employees page: <a href='" + department_address + "'>redirect</a>")
        }});
    let send_btn = $("#insert_btn")


    // Delete button function with ajax query
    $('#delete_button').click(function () {
        let data = JSON.stringify({'data': [department_id]})
        $.ajax({
            url: "/departments_crud",
            dataType: "json",
            data: data,
            'contentType': "application/json; charset=UTF-8",
            type: "DELETE",
            success: function (response) {
                console.log(response.responseText),
                $("#all_content").html("Department has been deleted. Open departments page: <a href='" + department_address + "'>redirect</a>")

            },
            error: function (response) {
                alert(response.responseText)
            }
        })
    });

    // Update button function
    $('#update_button').click(function () {
        $("#insert_btn").attr("id", "send_update");
        $("#form_title").text('Enter new values for this department')
    });

    // Send insert/update ajax query
    $('#insert_form').submit(function (event) {
        event.preventDefault()
        let form = $("#insert_form").serialize();
        if (send_btn.attr("id") === "insert_btn") {
            // Insert query
            $.ajax({
                url: "/departments_crud",
                dataType: "json",
                data: form,
                type: "POST",
                success: function (response) {
                    alert(response['Message'])
                },
                error: function (response) {
                    alert(response.responseText)
                }
            })
        } else {
            // Update query
            $.ajax({
                url: "/departments_crud",
                dataType: "json",
                data: form,
                type: "PUT",
                success: function (response) {
                    alert(response)
                    $("#send_update").attr("id", "insert_btn");
                    $("#form_title").text('Create a copy of the department ' + department_name)
                },
                error: function (response) {
                    alert(response.responseText)
                }
            })
        }
    });
});