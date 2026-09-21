function inputAccordionChecked(id, checked) {
    let accordion = gradioApp().getElementById(id);
    if (!accordion?.visibleCheckbox) {
        return;
    }
    accordion.visibleCheckbox.checked = checked;
    accordion.onVisibleCheckboxChange();
}

function setupAccordion(accordion) {
    let labelWrap = accordion.querySelector(".label-wrap");
    // Gradio 5 may render the element with the id on the input itself, or on
    // a wrapper. Support both layouts so the visible toggle can be created.
    let checkboxHost = gradioApp().getElementById(accordion.id + "-checkbox");
    let gradioCheckbox = checkboxHost?.matches("input[type=checkbox]")
        ? checkboxHost
        : checkboxHost?.querySelector("input[type=checkbox]");
    gradioCheckbox ??= gradioApp().querySelector(
        "#" + accordion.id + "-checkbox input[type=checkbox]",
    );
    let extra = gradioApp().querySelector("#" + accordion.id + "-extra");
    let span = labelWrap.querySelector("span");
    let linked = true;

    let isOpen = function () {
        return labelWrap.classList.contains("open");
    };

    let observerAccordionOpen = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutationRecord) {
            accordion.classList.toggle("input-accordion-open", isOpen());

            if (linked) {
                accordion.visibleCheckbox.checked = isOpen();
                accordion.onVisibleCheckboxChange();
            }
        });
    });
    observerAccordionOpen.observe(labelWrap, {
        attributes: true,
        attributeFilter: ["class"],
    });

    if (extra) {
        labelWrap.insertBefore(extra, labelWrap.lastElementChild);
    }

    accordion.onChecked = function (checked) {
        if (isOpen() != checked) {
            labelWrap.click();
        }
    };

    let visibleCheckbox = document.createElement("INPUT");
    visibleCheckbox.type = "checkbox";
    visibleCheckbox.checked = isOpen();
    visibleCheckbox.id = accordion.id + "-visible-checkbox";
    visibleCheckbox.className =
        (gradioCheckbox?.className || "") + " input-accordion-checkbox";
    span.insertBefore(visibleCheckbox, span.firstChild);

    accordion.visibleCheckbox = visibleCheckbox;
    accordion.onVisibleCheckboxChange = function () {
        if (linked && isOpen() != visibleCheckbox.checked) {
            labelWrap.click();
        }

        if (gradioCheckbox) {
            // Use the native click so Gradio receives its normal change event
            // and includes the value in the generation request.
            if (gradioCheckbox.checked != visibleCheckbox.checked) {
                gradioCheckbox.click();
            }
        }
    };

    visibleCheckbox.addEventListener("click", function (event) {
        linked = false;
        event.stopPropagation();
    });
    visibleCheckbox.addEventListener("input", accordion.onVisibleCheckboxChange);
}

onUiLoaded(function () {
    for (let accordion of gradioApp().querySelectorAll(".input-accordion")) {
        setupAccordion(accordion);
    }
});
