document.onkeydown = keyPressed;
document.onload = init_flipper;
currFrame = 0;
prevFrame = 0;
variantIndex = 0;
numFrames = frames.length;
frameEnabled = new Array(numFrames);

function toggleHelp() {
    $('#help').toggle();
}

function init_flipper(e) {
    genFrameList();
    gotoFrame(0);
    selectVariant(0);
    hideDataTable();
    $('#title').html(title);
    $('title').text("Image Flipper: " + title);
}

function toggleEnabled(index) {
    frameEnabled[index] = !frameEnabled[index];
    if (frameEnabled[index]) {
        $('#g' + index).removeClass('disabled');
    }
    else {
        $('#g' + index).addClass('disabled');
    }
}

function enableHelper(op) {
    for (var i = 0; i < frameEnabled.length; i++) {
        frameEnabled[i] = op(frameEnabled[i]);
        if (frameEnabled[i]) $('#g' + i).removeClass('disabled');
        else                 $('#g' + i).addClass('disabled');
    }
}

function enableAll()  { enableHelper(function(bool) { return true; }); }
function disableAll() { enableHelper(function(bool) { return false; }); }
function toggleAll()  { enableHelper(function(bool) { return !bool; }); }

function gotoFrame(index) {
    $('#g' + prevFrame).removeClass('previous selected');
    if (currFrame != index) {
        $('#g' + currFrame).removeClass('selected');
        $('#g' + currFrame).addClass('previous');
    }
    $('#g' + index).addClass('selected');
    if (currFrame != index) {
        prevFrame = currFrame;
        currFrame = index;
    }

    frame = frames[currFrame];
    $('#img_frame').attr({src: frame.image[variantIndex]});
    dataString = "<table>";

    hasData = false;
    for (var i = 0; i < statistics.length; i++) {
        stat = statistics[i];
        value = (stat in frame) ? frame[stat] : '----';
        hasData |= (stat in frame);
        dataString += "<tr><th>" + stat + "</th><td>" + value + "</td></tr>";
    }

    if (hasData) { dataString += "</table>" }
    else { dataString = "(No metadata)" }
    
    $('#frameData').html(dataString);
}

function keyPressed(e) {
    e = e || window.event;
    newFrame = currFrame;
    if (e.keyCode == '37') {
        // left arrow returns to previous (enabled) frame
        do {
            --newFrame;
            if (newFrame < 0) newFrame = numFrames - 1;
        } while (!frameEnabled[newFrame] && (newFrame != currFrame));
    }
    else if (e.keyCode == '39') {
        // right arrow advances the frame
        do {
            ++newFrame;
            if (newFrame >= numFrames) newFrame = 0;
        } while (!frameEnabled[newFrame] && (newFrame != currFrame));
    }
    else if (e.keyCode == '32') {
        // space bar toggles between frames
        newFrame = prevFrame;
        e.preventDefault();
    }

    else if (e.keyCode == '69') {
        // e key enables all
        enableAll();
    }

    else if (e.keyCode == '68') {
        // d key disables all
        disableAll();
    }

    else if (e.keyCode == '84') {
        // t key toggles all
        toggleAll();
    }

    if (newFrame != currFrame) {
        gotoFrame(newFrame);
    }
}

function genFrameList() {
    var strList = [];
    for (var i = 0; i < frames.length; i++) {
        strList.push("<a href='javascript:gotoFrame(" + i + ")' id='g"
                + i + "' oncontextmenu='javascript:toggleEnabled(" + i + "); return false;'>"
                + (i + 1) + "</a>");
        frameEnabled[i] = true;
    }
    $('#frameList').html(strList.join(''));
}

function selectVariant(index) {

    var strList = [];
    variantIndex = index;
    for (var i = 0; i < variants.length; i++) {
        classString = (i == variantIndex) ? 'class = "selected" ' : '';
        strList.push("<a " + classString + "href='javascript:selectVariant(" + i + ")'>"
                + variants[i] + "</a>");
    }
    if (variantIndex == index) {
        // Refresh frame if the variant changed
        gotoFrame(currFrame);
    }
    $('#variants').html(strList.join(''));
}

function showDataTable() {
    var strList = [];
    strList.push("<a href='javascript:hideDataTable()'>-</a><b>Source Data</b><pre>");
    strList.push("frame\t" + statistics.join("\t") + "\n");

    for (var i = 0; i < frames.length; i++) {
        frame = frames[i];
        var lineItems = [];
        for (var j = 0; j < statistics.length; j++) {
            stat = statistics[j];
            lineItems.push(frame[stat]);
        }
        strList.push((i + 1) + "\t" + lineItems.join("\t") + "\n");
    }
    strList.push("</pre>");
    $('#dataTable').html(strList.join(''));
}

function hideDataTable() {
    $('#dataTable').html("<a href='javascript:showDataTable()'>+</a><b>Source Data</b>");
}
