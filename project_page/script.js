const unit = 10;
const object_list = ['quad', 'fish', 'glasses']

// document.addEventListener('DOMContentLoaded', () => {

console.log('this is first');

const modelViewer1 = document.getElementById('model1');
const modelViewer2 = document.getElementById('model2');
const modelViewer3 = document.getElementById('model3');
const modelViewer4 = document.getElementById('model4');

const observer1 = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
        const realSrc = modelViewer1.src;
        modelViewer1.setAttribute('src', realSrc);
        observer1.unobserve(modelViewer1);
    }
});
const observer2 = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
        const realSrc = modelViewer2.src;
        modelViewer2.setAttribute('src', realSrc);
        observer2.unobserve(modelViewer2);
    }
});
const observer3 = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
        const realSrc = modelViewer3.src;
        modelViewer3.setAttribute('src', realSrc);
        observer3.unobserve(modelViewer3);
    }
});
const observer4 = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
        const realSrc = modelViewer4.src;
        modelViewer4.setAttribute('src', realSrc);
        observer4.unobserve(modelViewer4);
    }
});

observer1.observe(modelViewer1);
observer1.observe(modelViewer2);
observer1.observe(modelViewer3);
observer1.observe(modelViewer4);

observer2.observe(modelViewer1);
observer2.observe(modelViewer2);
observer2.observe(modelViewer3);
observer2.observe(modelViewer4);

observer3.observe(modelViewer1);
observer3.observe(modelViewer2);
observer3.observe(modelViewer3);
observer3.observe(modelViewer4);

observer4.observe(modelViewer1);
observer4.observe(modelViewer2);
observer4.observe(modelViewer3);
observer4.observe(modelViewer4);

    // modelViewer1.addEventListener('load', function () {
    //     const threeCamera = modelViewer1.getCameraOrbit();
    //     const fov = 17.6;

    //     // modelViewer1.fieldOfView = fov;
    //     // modelViewer1.camera.fieldOfView = fov;
    //     // modelViewer1.camera.updateProjectionMatrix();
    // });

    // modelViewer2.addEventListener('load', function () {
    //     const threeCamera = modelViewer2.getCameraOrbit();
    //     const fov = 17.6;

    //     // modelViewer2.fieldOfView = fov;
    //     // modelViewer2.camera.updateProjectionMatrix();
    // });

    // modelViewer3.addEventListener('load', function () {
    //     const threeCamera = modelViewer3.getCameraOrbit();
    //     const fov = 17.6;

    //     // modelViewer3.fieldOfView = fov;
    //     // modelViewer3.camera.updateProjectionMatrix();
    // });

    // modelViewer4.addEventListener('load', function () {
    //     const threeCamera = modelViewer4.getCameraOrbit();
    //     const fov = 17.6;

    //     // modelViewer4.fieldOfView = fov;
    //     // modelViewer4.camera.updateProjectionMatrix();
    // });
// });

const model1 = document.getElementById('model1');
const model2 = document.getElementById('model2');
const model3 = document.getElementById('model3');
const model4 = document.getElementById('model4');

let syncing = false;

let syncTimeout1 = null;
let syncTimeout2 = null;
let syncTimeout3 = null;
let syncTimeout4 = null;

const SYNC_DELAY_MS = 50;

function restartAutoRotate () {

    // do nothing because we cannot detect if user has interacted with viewer and constant rotation is fraustrating
    return

    const modelViewer1 = document.getElementById('model1');
    const modelViewer2 = document.getElementById('model2');
    const modelViewer3 = document.getElementById('model3');
    const modelViewer4 = document.getElementById('model4');

    modelViewer1.setAttribute('auto-rotate', '');
    modelViewer2.setAttribute('auto-rotate', '');
    modelViewer3.setAttribute('auto-rotate', '');
    modelViewer4.setAttribute('auto-rotate', '');

    console.log('restarted');

    modelViewer1.addEventListener('user-gesture', () => {
        console.log('interaction detected');
        modelViewer1.removeAttribute('auto-rotate');
    });
    modelViewer2.addEventListener('user-gesture', () => {
        modelViewer2.removeAttribute('auto-rotate');
    });
    modelViewer3.addEventListener('user-gesture', () => {
        modelViewer3.removeAttribute('auto-rotate');
    });
    modelViewer4.addEventListener('user-gesture', () => {
        modelViewer4.removeAttribute('auto-rotate');
    });
}

// copy model1 camera to 2,3,4 
model1.addEventListener('camera-change', () => {
    if (!syncing) {

        if (syncTimeout1) {
            clearTimeout(syncTimeout1);
        }

        syncTimeout1 = setTimeout(() => {
            syncing = true;
            copyCameraSettings(model1, model2);
            copyCameraSettings(model1, model3);
            copyCameraSettings(model1, model4);
            requestAnimationFrame(() => {
                syncing = false;
            });
        }, SYNC_DELAY_MS);
    }
});

model2.addEventListener('camera-change', () => {
    if (!syncing) {
        if (syncTimeout2) {
            clearTimeout(syncTimeout2);
        }
        syncTimeout2 = setTimeout(() => {
            syncing = true;
            copyCameraSettings(model2, model1);
            copyCameraSettings(model2, model3);
            copyCameraSettings(model2, model4);
            requestAnimationFrame(() => {
                syncing = false;
            });
        }, SYNC_DELAY_MS);
    }
});

model3.addEventListener('camera-change', () => {
    if (!syncing) {
        if (syncTimeout3) {
            clearTimeout(syncTimeout3);
        }
        syncTimeout3 = setTimeout(() => {
            syncing = true;
            copyCameraSettings(model3, model1);
            copyCameraSettings(model3, model2);
            copyCameraSettings(model3, model4);
            requestAnimationFrame(() => {
                syncing = false;
            });
        }, SYNC_DELAY_MS);
    }
});

model4.addEventListener('camera-change', () => {
    if (!syncing) {
        if (syncTimeout4) {
            clearTimeout(syncTimeout4);
        }
        syncTimeout4 = setTimeout(() => {
            syncing = true;
            copyCameraSettings(model4, model1);
            copyCameraSettings(model4, model2);
            copyCameraSettings(model4, model3);
            requestAnimationFrame(() => {
                syncing = false;
            });
        }, SYNC_DELAY_MS);
    }
});

function copyCameraSettings(sourceModel, targetModel) {
    const orbit = sourceModel.getCameraOrbit();
    const fieldOfView = sourceModel.getFieldOfView();
    const target = sourceModel.getCameraTarget();
    
    targetModel.cameraOrbit = `${orbit.theta}rad ${orbit.phi}rad ${orbit.radius}m`;
    targetModel.fieldOfView = `${fieldOfView}deg`;
    targetModel.cameraTarget = `${target.x}m ${target.y}m ${target.z}m`;

    console.log('Camera Orbit:', targetModel.cameraOrbit);
    console.log('Camera FOV:', targetModel.fieldOfView);
    console.log('Camera Target:', targetModel.cameraTarget);
}


// Mesh indexing and loading
function loadNewMesh(model_id, path) {
    const viewer = document.getElementById(model_id);
    viewer.setAttribute('src', path);
}

// use object-id, slider-0, slider-1, slider-2 to index
let object_id = object_list[0];  // by default
let slider_value_0 = 0;
let slider_value_1 = 0;
let slider_value_2 = 0;

const scrollContainer = document.querySelector('.scroll-container');
const sections = document.querySelectorAll(".section");
const dots = document.querySelectorAll(".section-nav .dot");
let current = "cover";
let chromeVisible = true;

function setChromeVisibility(show) {
    if (chromeVisible === show) {
        return;
    }
    chromeVisible = show;

    const opacity = show ? '1' : '0';
    const pointerEvents = show ? 'auto' : 'none';
    const menuFooter = document.getElementById('menu-footer');
    const topBar = document.getElementById('top-bar');
    const mobileMenu = document.getElementById('mobile-menu');
    const mobileTopBar = document.getElementById('mobile-top-bar');

    [menuFooter, topBar, mobileMenu, mobileTopBar].forEach((el) => {
        if (!el) return;
        el.style.opacity = opacity;
        el.style.pointerEvents = pointerEvents;
        el.style.transition = 'opacity 0.3s ease';
    });
}

let scrollTicking = false;

function handleScroll() {
    if (!scrollContainer) {
        return;
    }
    scrollTicking = false;

    // Loop through each section to find the one in view
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionBottom = sectionTop + section.offsetHeight;
        const scrollPosition = scrollContainer.scrollTop + 0.5 * scrollContainer.clientHeight;

        // console.log('id', section.getAttribute("id"), 'section top', sectionTop, 'section bottom', sectionBottom, 'scroll position', scrollPosition);

        // Check if the section is in the viewport
        if (scrollPosition > sectionTop && scrollPosition <= sectionBottom) {
            current = section.getAttribute("id");
        }
    });

    // Loop through each dot to set the active class
    dots.forEach(dot => {
        dot.classList.remove("active");
        // Match the dot's href to the current section id
        if (dot.getAttribute("href").slice(1) === current) {
            dot.classList.add("active");
        }
    });

    setChromeVisibility(current === 'cover');
}

// Slider control
if (scrollContainer) {
    scrollContainer.addEventListener("scroll", () => {
        if (scrollTicking) {
            return;
        }
        scrollTicking = true;
        requestAnimationFrame(handleScroll);
    }, { passive: true });
    handleScroll();
}

// return mesh paths for all 4 viewers, indexed by global vars
function getMeshPaths() {
    if (object_id === 'quad') {
        const [slider_0, slider_1, slider_2] = [
            String(slider_value_0).padStart(2, '0'),
            String(slider_value_1).padStart(2, '0'), 
            String(slider_value_2).padStart(2, '0')
        ];
        // console.log('here');
        return [
            'assets/meshes/quad/mesh/00_00_00.glb',
            'assets/meshes/quad/blob/00_00_00.glb',
            'assets/meshes/quad/blob/' + slider_0 + '_' + slider_1 + '_' + slider_2 + '.glb',
            'assets/meshes/quad/mesh/' + slider_0 + '_' + slider_1 + '_' + slider_2 + '.glb',
        ]
    } else if (object_id === 'fish') {
        console.log('fish is being clicked');
        const [slider_0, slider_1, slider_2] = [
            String(slider_value_0).padStart(2, '0'),
            String(slider_value_1).padStart(2, '0'), 
            String(slider_value_2).padStart(2, '0')
        ];
        return [
            'assets/meshes/fish/mesh/00_00_00.glb',
            'assets/meshes/fish/blob/00_00_00.glb',
            'assets/meshes/fish/blob/' + slider_0 + '_' + slider_1 + '_' + slider_2 + '.glb',
            'assets/meshes/fish/mesh/' + slider_0 + '_' + slider_1 + '_' + slider_2 + '.glb',
        ]
    } else if (object_id === 'glasses') {
        console.log('glasses is being clicked');
        const [slider_0, slider_1, slider_2] = [
            String(slider_value_0).padStart(2, '0'),
            String(slider_value_1).padStart(2, '0'), 
            String(slider_value_2).padStart(2, '0')
        ];
        return [
            'assets/meshes/glasses/mesh/00_00_00.glb',
            'assets/meshes/glasses/blob/00_00_00.glb',
            'assets/meshes/glasses/blob/' + slider_0 + '_' + slider_1 + '_' + slider_2 + '.glb',
            'assets/meshes/glasses/mesh/' + slider_0 + '_' + slider_1 + '_' + slider_2 + '.glb',
        ]
    } else {
        return [
            'assets/meshes/example/mesh.glb',
            'assets/meshes/example/mesh.glb',
            'assets/meshes/example/mesh.glb',
            'assets/meshes/example/mesh.glb',
        ]
    }
}

// reload mesh when object selection or slider changes
async function handleObjectRefresh(reload_all) {
    // let viewer_0_path = '';
    // let viewer_1_path = '';
    // let viewer_2_path = '';
    // let viewer_3_path = '';

    const [viewer_0_path, viewer_1_path, viewer_2_path, viewer_3_path] = getMeshPaths();

    // console.log('reload all', reload_all, 'viewer_0_path', viewer_0_path, 'viewer_1_path', viewer_1_path, 'viewer_2_path', viewer_2_path, 'viewer_3_path', viewer_3_path);

    if (reload_all) {
        restartAutoRotate();
        await Promise.all([
            loadNewMesh('model1', viewer_0_path),
            loadNewMesh('model2', viewer_1_path),
            loadNewMesh('model3', viewer_2_path),
            loadNewMesh('model4', viewer_3_path),
        ]);
        console.log("reload all complete");
        setCamera();
    } else {
        await Promise.all([
            loadNewMesh('model3', viewer_2_path),
            loadNewMesh('model4', viewer_3_path),
        ]);
        console.log("reload one complete");

    }
}


function handleSliderValueChange(sliderId, value_fp) {
    console.log('sliderId', sliderId, 'value_fp', value_fp);

    let value = 0.0;
    if (object_id === 'glasses' && sliderId === 'slider2') {  // This is a special case
        value = Math.max(Math.floor(value_fp * (8 - 0.01)), 0);
    } else {
        value = Math.max(Math.floor(value_fp * (unit - 0.01)), 0);
    }

    if (sliderId === 'slider1' && value !== slider_value_0) {
        slider_value_0 = value;
        handleObjectRefresh(reload_all=false);
    } else if (sliderId === 'slider2' && value !== slider_value_1) {
        slider_value_1 = value;
        handleObjectRefresh(reload_all=false);
    } else if (sliderId === 'slider3' && value !== slider_value_2) {
        slider_value_2 = value;
        handleObjectRefresh(reload_all=false);
    }         
}

function setupSlider(sliderId) {
    var slider = document.getElementById(sliderId);
    var thumb = slider.querySelector('.thumb');
    var isDragging = false;
    let debounceTimer = null;
    var value = 0;

    var initialTop = slider.offsetHeight - thumb.offsetHeight;
    thumb.style.top = initialTop + 'px';

    window.addEventListener('resize', function () {
        console.log('resize detected');
        const sliderHeight = slider.offsetHeight;
        const thumbHeight = thumb.offsetHeight;
        const newTop = (1 - value) * (sliderHeight - thumbHeight);
        thumb.style.top = `${newTop}px`;
    });

    // Mouse down event to start dragging
    thumb.addEventListener('mousedown', function (e) {
        e.preventDefault(); // Prevent text selection
        isDragging = true;
        var offset = e.clientY - thumb.offsetTop;

        document.onmousemove = function (e) {
            e.preventDefault(); // Prevent text selection
            if (isDragging) {
                var newTop = e.clientY - offset;
                var sliderHeight = slider.offsetHeight;
                var thumbHeight = thumb.offsetHeight;

                newTop = Math.max(0, Math.min(sliderHeight - thumbHeight, newTop));
                thumb.style.top = newTop + 'px';

                value = 1 - newTop / (sliderHeight - thumbHeight); // value in range [0, 1]
                // console.log(`Slider ${sliderId} value: ${value.toFixed(2)}`);
                // call slider value-change handle

                // avoid debouncing
                clearTimeout(debounceTimer);
                DEBOUNCE_TIME = 0;
                // handleSliderValueChange(sliderId, value);
                debounceTimer = setTimeout(() => {
                    handleSliderValueChange(sliderId, value);
                }, DEBOUNCE_TIME);
            }
        };

        document.onmouseup = function () {
            isDragging = false;
        };
    });
}

// $(document).ready(function () {
    // Initialize all sliders
setupSlider('slider1');
setupSlider('slider2');
setupSlider('slider3');

// for some reason the camera would re-init the first time slide is triggered, 
// so we trigger it the first time before the user can change the slider during the init stage 

// handleObjectRefresh(reload_all=false);

    // handleSliderValueChange('slider1', 0);
    // handleSliderValueChange('slider2', 0);
    // handleSliderValueChange('slider3', 0);
// });

// TODO: set camera according to object-id
// set camera is only called after changing object or right after initialization
function setCamera() {

    let camera_orbit = "-2.4272486433837064rad 1.286506155631572rad 1.893689527071967m";
    
    if (object_id === 'quad') {
        camera_orbit = "-2.4272486433837064rad 1.286506155631572rad 1.893689527071967m";
        console.log('bear camera set');
    } else if (object_id === 'fish') {
        console.log('fish camera set');
        camera_orbit = "0.9096008991865057rad 0.6936332515421794rad 1.5335984860795575m";
    } else if (object_id === 'glasses') {
        console.log('glasses camera set');
        camera_orbit = "0.7343012552040333rad 1.0732983208709015rad 1.719736902372597m";
    }else {
        console.log('unknown camera set');
        camera_orbit = "-2.4272486433837064rad 1.286506155631572rad 1.893689527071967m";
    }

    const camera_target = "-0.m -0.m 0.m";
    // const camera_ori = "0 0 1 90deg";
    // const camera_fov = "30deg";

    const viewer_1 = document.getElementById('model1');
    const viewer_2 = document.getElementById('model2');
    const viewer_3 = document.getElementById('model3');
    const viewer_4 = document.getElementById('model4');

    viewer_1.setAttribute('camera-orbit', camera_orbit);
    viewer_2.setAttribute('camera-orbit', camera_orbit);
    viewer_3.setAttribute('camera-orbit', camera_orbit);
    viewer_4.setAttribute('camera-orbit', camera_orbit);

    viewer_1.setAttribute('camera-target', camera_target);
    viewer_2.setAttribute('camera-target', camera_target);
    viewer_3.setAttribute('camera-target', camera_target);
    viewer_4.setAttribute('camera-target', camera_target);

    console.log('set camera', viewer_1.getAttribute('camera-orbit'), viewer_1.getAttribute('camera-target'));

    // viewer_1.setAttribute('camera-fov', camera_fov);
    // viewer_2.setAttribute('camera-fov', camera_fov);
    // viewer_3.setAttribute('camera-fov', camera_fov);
    // viewer_4.setAttribute('camera-fov', camera_fov);
}

function handleCardClick(index) {
    console.log("Card clicked:", index);

    // modify object id if not the same
    const new_object_id = object_list[index];
    // if (new_object_id !== object_id) {
    object_id = new_object_id;

    // reset slider
    updateSlidersVisibility();  // Call this after
    document.querySelectorAll('.slider').forEach(slider => {
        const thumb = slider.querySelector('.thumb');
        const sliderHeight = slider.offsetHeight;
        const thumbHeight = thumb.offsetHeight;
        const newTop = sliderHeight - thumbHeight;
    
        console.log('reseting slider');
        thumb.style.top = newTop + 'px';
        // handleSliderValueChange(slider, 0);
    });

    slider_value_0 = 0;
    slider_value_1 = 0;
    slider_value_2 = 0;

    handleObjectRefresh(reload_all=true);

    // reset camera
    setCamera();

    // highlight the selection
    const cards = document.querySelectorAll(".selection-card");

    // Then remove 'selected' from all other cards
    cards.forEach((card, i) => {
      if (i !== index) {
        card.classList.remove("selected");
      }
    });
    cards[index].classList.add("selected");

}

// object selection logic
const cards = document.querySelectorAll(".selection-card");
cards.forEach((card, index) => {
    card.addEventListener("click", () => {
        handleCardClick(index);
    });
});

// hide 3rd slider for glasses
function updateSlidersVisibility() {
    const slider3 = document.getElementById('slider3');
    console.log('update slider visibility', object_id);
    if (object_id === 'glasses') {
        slider3.style.display = 'none';
    } else {
        slider3.style.display = 'block';
    }
}

// load for the first time
handleObjectRefresh(reload_all=true);

setCamera();

model1.addEventListener('camera-change', () => {
    console.log('Camera Orbit:', model1.getAttribute('camera-orbit'));
    console.log('Camera Target:', model1.getAttribute('camera-target'));
});

model1.addEventListener('load', setCamera)


// For hero icon animation
const icons = [
    "ti ti-dog",
    "ti ti-fridge",
    "ti ti-cat",
    "ti ti-scissors",
    "ti ti-fish",
    "ti ti-scissors",
    "ti ti-mood-happy",
    "ti ti-pig",
  ];

let current_icon_id = 0;
const iconElement = document.getElementById("changing-icon");

function updateIcon() {
    console.log('Changing icon')
    iconElement.classList.add("fade");
    setTimeout(() => {
      iconElement.className = icons[current_icon_id];
      iconElement.classList.remove("fade");
      current_icon_id = (current_icon_id + 1) % icons.length;
    }, 100);
  }

// Initial icon
updateIcon();

// Change every 1000ms (1 second)
setInterval(updateIcon, 1000);

// Pre-fetch all meshes into cache (do this in order)
async function preloadMeshes(meshPaths, cacheName = 'mesh-cache-v1') {
    const cache = await caches.open(cacheName);
    for (const path of meshPaths) {
        console.log(`Preloading mesh: ${path}`);
        const response = await fetch(path);
        if (response.ok) {
        await cache.put(path, response.clone());
        } else {
        console.warn(`Failed to preload: ${path}`);
        }
    }
}

function generateMeshPaths(baseDir, dim1, dim2, dim3) {
    const paths = [];
    // Let larger digits load first
    for (let k = 0; k < dim3; k++) {
      for (let j = 0; j < dim2; j++) {
        for (let i = 0; i < dim1; i++) {
          const filename = `${String(i).padStart(2, '0')}_${String(j).padStart(2, '0')}_${String(k).padStart(2, '0')}.glb`;
          // Make sure baseDir ends with '/' or add it
          const fullPath = baseDir.endsWith('/') ? baseDir + filename : baseDir + '/' + filename;
          paths.push(fullPath);
        }
      }
    }
    return paths;
}

function mergeClose(list1, list2) {
    if (list1.length !== list2.length) {
      throw new Error('Both lists must be the same length');
    }
    
    const merged = [];
    for (let i = 0; i < list1.length; i++) {
      merged.push(list1[i]);
      merged.push(list2[i]);
    }
    return merged;
  }

async function prefetchMeshes() {
    const isTouchDevice = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;
    const isMobileUA = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent || '');
    const hasLowDeviceMemory = typeof navigator.deviceMemory === 'number' && navigator.deviceMemory <= 4;

    // On mobile/low-memory devices this can aggressively consume memory and crash Safari tabs.
    if (isTouchDevice || isMobileUA || hasLowDeviceMemory) {
        console.log('Skipping mesh prefetch on mobile or low-memory device');
        return;
    }

    if (!('caches' in window) || typeof caches.open !== 'function') {
        return;
    }

    const quad_mesh_paths = generateMeshPaths("assets/meshes/quad/mesh", 10, 10, 10);
    // console.log('qua mesdh paths', quad_mesh_paths);
    const quad_blob_paths = generateMeshPaths("assets/meshes/quad/blob", 10, 10, 10);
    const quad_paths = mergeClose(quad_blob_paths, quad_mesh_paths);
    await preloadMeshes(quad_paths, 'mesh-cache-quad');  // Blobs are closer to user's interactions so we preload them first

    const fish_mesh_paths = generateMeshPaths("assets/meshes/fish/mesh", 10, 10, 10);
    const fish_blob_paths = generateMeshPaths("assets/meshes/fish/blob", 10, 10, 10);
    const fish_paths = mergeClose(fish_blob_paths, fish_mesh_paths);
    await preloadMeshes(fish_paths, 'mesh-cache-fish');

    const glasses_mesh_paths = generateMeshPaths("assets/meshes/glasses/mesh", 10, 10, 1);
    const glasses_blob_paths = generateMeshPaths("assets/meshes/glasses/blob", 10, 10, 1);
    const glasses_paths = mergeClose(glasses_blob_paths, glasses_mesh_paths);
    await preloadMeshes(glasses_paths, 'mesh-cache-glasses');
}

prefetchMeshes();
