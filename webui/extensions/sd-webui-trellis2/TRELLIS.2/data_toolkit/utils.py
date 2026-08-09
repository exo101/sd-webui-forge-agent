from typing import *
import hashlib
import numpy as np
import cv2


def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    # Read the file from the path
    with open(file, "rb") as f:
        # Update the hash with the file content
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

# ===============LOW DISCREPANCY SEQUENCES================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

# ==============PLY IO===============
import struct
import re
import torch

def read_ply(filename):
    """
    Read a PLY file and return vertices, triangle faces, and quad faces.
    
    Args:
        filename (str): The file path to read from.
        
    Returns:
        vertices (torch.Tensor): Tensor of shape [N, 3] containing vertex positions.
        tris (torch.Tensor): Tensor of shape [M, 3] containing triangle face indices (empty if none).
        quads (torch.Tensor): Tensor of shape [K, 4] containing quad face indices (empty if none).
    """
    with open(filename, 'rb') as f:
        # Read the header until 'end_header' is encountered
        header_bytes = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header not found")
            header_bytes += line
            if b"end_header" in line:
                break
        header = header_bytes.decode('utf-8')
        
        # Determine if the file is in ASCII or binary format
        is_ascii = "ascii" in header
        
        # Extract the number of vertices and faces from the header using regex
        vertex_match = re.search(r'element vertex (\d+)', header)
        if vertex_match:
            num_vertices = int(vertex_match.group(1))
        else:
            raise ValueError("Vertex count not found in header")
            
        face_match = re.search(r'element face (\d+)', header)
        if face_match:
            num_faces = int(face_match.group(1))
        else:
            raise ValueError("Face count not found in header")
        
        vertices = []
        tris = []
        quads = []
        
        if is_ascii:
            # For ASCII format, read each line of vertex data (each line contains 3 floats)
            for _ in range(num_vertices):
                line = f.readline().decode('utf-8').strip()
                if not line: 
                    continue
                parts = line.split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
            # Read face data, where the first number indicates the number of vertices for the face
            for _ in range(num_faces):
                line = f.readline().decode('utf-8').strip()
                if not line: 
                    continue
                parts = line.split()
                count = int(parts[0])
                indices = list(map(int, parts[1:]))
                if count == 3:
                    tris.append(indices)
                elif count == 4:
                    quads.append(indices)
                else:
                    # Skip faces with other numbers of vertices (can be extended as needed)
                    pass
        else:
            # For binary format: read directly from the binary stream
            # Each vertex consists of 3 floats (12 bytes per vertex)
            for _ in range(num_vertices):
                data = f.read(12)
                if len(data) < 12:
                    raise ValueError("Insufficient vertex data")
                v = struct.unpack('<fff', data)
                vertices.append(v)
            
            # Read face data from the binary stream
            for _ in range(num_faces):
                # First, read 1 byte indicating the number of vertices in the face
                count_data = f.read(1)
                if len(count_data) < 1:
                    raise ValueError("Failed to read face vertex count")
                count = struct.unpack('<B', count_data)[0]
                if count == 3:
                    data = f.read(12)  # 3 * 4 bytes
                    if len(data) < 12:
                        raise ValueError("Insufficient data for triangle face")
                    indices = struct.unpack('<3i', data)
                    tris.append(indices)
                elif count == 4:
                    data = f.read(16)  # 4 * 4 bytes
                    if len(data) < 16:
                        raise ValueError("Insufficient data for quad face")
                    indices = struct.unpack('<4i', data)
                    quads.append(indices)
                else:
                    # For faces with a different number of vertices, read count*4 bytes
                    data = f.read(count * 4)
                    # Skip or extend processing as needed
                    raise ValueError(f"Unsupported face with {count} vertices")
        
        # Convert lists to torch.Tensor
        vertices = torch.tensor(vertices, dtype=torch.float32)
        tris = torch.tensor(tris, dtype=torch.int32) if len(tris) > 0 else torch.empty((0, 3), dtype=torch.int32)
        quads = torch.tensor(quads, dtype=torch.int32) if len(quads) > 0 else torch.empty((0, 4), dtype=torch.int32)
        
        return vertices, tris, quads


def write_ply(filename, vertices, tris, quads, ascii=False):
    """
    Write a mesh to a PLY file, with the option to save in ASCII or binary format.
    
    Args:
        filename (str): The filename to write to.
        vertices (torch.Tensor): [N, 3] The vertex positions.
        tris (torch.Tensor): [M, 3] The triangle indices.
        quads (torch.Tensor): [K, 4] The quad indices.
        ascii (bool): If True, write in ASCII format. If False, write in binary format.
    """
    # Convert torch tensors to numpy arrays
    vertices = vertices.numpy()
    tris = tris.numpy()
    quads = quads.numpy()

    # Prepare the header
    num_vertices = len(vertices)
    num_faces = len(tris) + len(quads)

    # Vertex properties
    vertex_header = "property float x\nproperty float y\nproperty float z"

    # Face properties (the number of vertices per face is variable)
    face_header = "property list uchar int vertex_index"

    # Start writing the PLY header
    header = f"ply\n"
    header += f"format {'ascii 1.0' if ascii else 'binary_little_endian 1.0'}\n"
    header += f"element vertex {num_vertices}\n"
    header += vertex_header + "\n"
    header += f"element face {num_faces}\n"
    header += face_header + "\n"
    header += "end_header\n"

    # Open the file for writing
    with open(filename, 'wb' if not ascii else 'w') as f:
        # Write the header
        f.write(header if ascii else header.encode('utf-8'))

        # Write the vertex data
        if ascii:
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
        else:
            for v in vertices:
                f.write(struct.pack('<fff', *v))

        # Write the face data
        if ascii:
            for tri in tris:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
            for quad in quads:
                f.write(f"4 {quad[0]} {quad[1]} {quad[2]} {quad[3]}\n")
        else:
            for tri in tris:
                f.write(struct.pack('<B3i', 3, *tri))  # 3 indices for triangle
            for quad in quads:
                f.write(struct.pack('<B4i', 4, *quad))  # 4 indices for quad
                
                
# ==============IMAGE UTILS===============

def make_grid(images, nrow=None, ncol=None, aspect_ratio=None):
    num_images = len(images)
    if nrow is None and ncol is None:
        if aspect_ratio is not None:
            nrow = int(np.round(np.sqrt(num_images / aspect_ratio)))
        else:
            nrow = int(np.sqrt(num_images))
        ncol = (num_images + nrow - 1) // nrow
    elif nrow is None and ncol is not None:
        nrow = (num_images + ncol - 1) // ncol
    elif nrow is not None and ncol is None:
        ncol = (num_images + nrow - 1) // nrow
    else:
        assert nrow * ncol >= num_images, 'nrow * ncol must be greater than or equal to the number of images'
    
    if images[0].ndim == 2:
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1]), dtype=images[0].dtype)
    else:
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1], images[0].shape[2]), dtype=images[0].dtype)
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        grid[row * img.shape[0]:(row + 1) * img.shape[0], col * img.shape[1]:(col + 1) * img.shape[1]] = img
    return grid


def notes_on_image(img, notes=None):
    img = np.pad(img, ((0, 32), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if notes is not None:
        img = cv2.putText(img, notes, (0, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



def text_image(text, resolution=(512, 512), max_size=0.5, h_align="left", v_align="center"):
    """
    Draw text on an image of the given resolution. The text is automatically wrapped
    and scaled so that it fits completely within the image while preserving any explicit
    line breaks and original spacing. Horizontal and vertical alignment can be controlled
    via flags.
    
    Parameters:
        text (str): The input text. Newline characters and spacing are preserved.
        resolution (tuple): The image resolution as (width, height).
        max_size (float): The maximum font size.
        h_align (str): Horizontal alignment. Options: "left", "center", "right".
        v_align (str): Vertical alignment. Options: "top", "center", "bottom".
        
    Returns:
        numpy.ndarray: The resulting image (BGR format) with the text drawn.
    """
    width, height = resolution
    # Create a white background image
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Set margins and compute available drawing area
    margin = 10
    avail_width = width - 2 * margin
    avail_height = height - 2 * margin

    # Choose OpenCV font and text thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    # Ratio for additional spacing between lines (relative to the height of "A")
    line_spacing_ratio = 0.5

    def wrap_line(line, max_width, font, thickness, scale):
        """
        Wrap a single line of text into multiple lines such that each line's
        width (measured at the given scale) does not exceed max_width.
        This function preserves the original spacing by splitting the line into tokens
        (words and whitespace) using a regular expression.
        
        Parameters:
            line (str): The input text line.
            max_width (int): Maximum allowed width in pixels.
            font (int): OpenCV font identifier.
            thickness (int): Text thickness.
            scale (float): The current font scale.
            
        Returns:
            List[str]: A list of wrapped lines.
        """
        # Split the line into tokens (words and whitespace), preserving spacing
        tokens = re.split(r'(\s+)', line)
        if not tokens:
            return ['']
        
        wrapped_lines = []
        current_line = ""
        for token in tokens:
            candidate = current_line + token
            candidate_width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
            if candidate_width <= max_width:
                current_line = candidate
            else:
                # If current_line is empty, the token itself is too wide;
                # break the token character by character.
                if current_line == "":
                    sub_token = ""
                    for char in token:
                        candidate_char = sub_token + char
                        if cv2.getTextSize(candidate_char, font, scale, thickness)[0][0] <= max_width:
                            sub_token = candidate_char
                        else:
                            if sub_token:
                                wrapped_lines.append(sub_token)
                            sub_token = char
                    current_line = sub_token
                else:
                    wrapped_lines.append(current_line)
                    current_line = token
        if current_line:
            wrapped_lines.append(current_line)
        return wrapped_lines

    def compute_text_block(scale):
        """
        Wrap the entire text (splitting at explicit newline characters) using the
        provided scale, and then compute the overall width and height of the text block.
        
        Returns:
            wrapped_lines (List[str]): The list of wrapped lines.
            block_width (int): Maximum width among the wrapped lines.
            block_height (int): Total height of the text block including spacing.
            sizes (List[tuple]): A list of (width, height) for each wrapped line.
            spacing (int): The spacing between lines (computed from the scaled "A" height).
        """
        # Split text by explicit newlines
        input_lines = text.splitlines() if text else ['']
        wrapped_lines = []
        for line in input_lines:
            wrapped = wrap_line(line, avail_width, font, thickness, scale)
            wrapped_lines.extend(wrapped)
            
        sizes = []
        for line in wrapped_lines:
            (text_size, _) = cv2.getTextSize(line, font, scale, thickness)
            sizes.append(text_size)  # (width, height)
            
        block_width = max((w for w, h in sizes), default=0)
        # Use the height of "A" (at the current scale) to compute line spacing
        base_height = cv2.getTextSize("A", font, scale, thickness)[0][1]
        spacing = int(line_spacing_ratio * base_height)
        block_height = sum(h for w, h in sizes) + spacing * (len(sizes) - 1) if sizes else 0
        
        return wrapped_lines, block_width, block_height, sizes, spacing

    # Use binary search to find the maximum scale that allows the text block to fit
    lo = 0.001
    hi = max_size
    eps = 0.001  # convergence threshold
    best_scale = lo
    best_result = None

    while hi - lo > eps:
        mid = (lo + hi) / 2
        wrapped_lines, block_width, block_height, sizes, spacing = compute_text_block(mid)
        # Ensure that both width and height constraints are met
        if block_width <= avail_width and block_height <= avail_height:
            best_scale = mid
            best_result = (wrapped_lines, block_width, block_height, sizes, spacing)
            lo = mid  # try a larger scale
        else:
            hi = mid  # reduce the scale

    if best_result is None:
        best_scale = 0.5
        best_result = compute_text_block(best_scale)
        
    wrapped_lines, block_width, block_height, sizes, spacing = best_result

    # Compute starting y-coordinate based on vertical alignment flag
    if v_align == "top":
        y_top = margin
    elif v_align == "center":
        y_top = margin + (avail_height - block_height) // 2
    elif v_align == "bottom":
        y_top = margin + (avail_height - block_height)
    else:
        y_top = margin + (avail_height - block_height) // 2  # default to center if invalid flag

    # For cv2.putText, the y coordinate represents the text baseline;
    # so for the first line add its height.
    y = y_top + (sizes[0][1] if sizes else 0)

    # Draw each line with horizontal alignment based on the flag
    for i, line in enumerate(wrapped_lines):
        line_width, line_height = sizes[i]
        if h_align == "left":
            x = margin
        elif h_align == "center":
            x = margin + (avail_width - line_width) // 2
        elif h_align == "right":
            x = margin + (avail_width - line_width)
        else:
            x = margin  # default to left if invalid flag

        cv2.putText(img, line, (x, y), font, best_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += line_height + spacing

    return img


def save_image_with_notes(img, path, notes=None):
    """
    Save an image with notes.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = notes_on_image(img, notes)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
