(function () {

    let isSetupForMobile = false;

    function isMobile() {
        for (let tab of ["txt2img", "img2img"]) {
            let imageTab = gradioApp().getElementById(tab + '_results');
            if (imageTab && imageTab.offsetParent && imageTab.offsetLeft == 0) {
                return true;
            }
        }

        return false;
    }

    function reportWindowSize() {
        // not applicable for compact prompt layout
        if (gradioApp().querySelector('.toprow-compact-tools')) return;

        let currentlyMobile = isMobile();
        if (currentlyMobile == isSetupForMobile) return;
        isSetupForMobile = currentlyMobile;

        for (let tab of ["txt2img", "img2img"]) {
            let button = gradioApp().getElementById(tab + '_generate_box');
            let target = gradioApp().getElementById(currentlyMobile ? tab + '_results' : tab + '_actions_column');
            target.insertBefore(button, target.firstElementChild);

            gradioApp().getElementById(tab + '_results').classList.toggle('mobile', currentlyMobile);
        }
    }

    window.addEventListener("resize", reportWindowSize);

    onUiLoaded(function () {
        reportWindowSize();
    });

})();
