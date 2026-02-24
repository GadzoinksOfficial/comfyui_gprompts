#!/usr/bin/env python3
"""
Immich Photo Import Tool
Import local photos to Immich server

Neal Katz,Deepseek and Claude
"""

import os
import sys
import argparse
import requests
import mimetypes
import hashlib
from pathlib import Path
from datetime import datetime
import time
import json

def dprint(s):
    #print(s)
    pass

class ImmichImporter:
    def __init__(self, server_url, api_key=None,importer_name = "ImmichImporter"):
        self.server_url = server_url.rstrip('/')
        self.api_base = f"{self.server_url}/api"
        dprint(f"self.api_base: {self.api_base}")
        self.api_key = api_key
        self.session = requests.Session()
        self.importer_name = importer_name
        
        if api_key:
            self.session.headers.update({'x-api-key': api_key})
            dprint(f"✓ Using API key for authentication")
    
    def test_connection(self):
        """Test connection to the server"""
        try:
            response = self.session.get(f"{self.api_base}/server/ping")
            if response.status_code == 200:
                dprint(f"✓ Connected to server at {self.server_url}")
                return True
        except requests.exceptions.RequestException as e:
            dprint(f"✗ Failed to connect to server: {e}")
            return False
    
    def login(self, email, password):
        """Login to get API key if not provided"""
        login_url = f"{self.api_base}/auth/login"
        payload = {
            'email': email,
            'password': password
        }
        
        try:
            response = self.session.post(login_url, json=payload)
            response.raise_for_status()
            data = response.json()
            self.api_key = data.get('accessToken')
            self.session.headers.update({'x-api-key': self.api_key})
            dprint("✓ Successfully logged in")
            dprint(f"  User: {data.get('userEmail')} ({data.get('name')})")
            return True
        except requests.exceptions.RequestException as e:
            dprint(f"✗ Login failed: {e}")
            if hasattr(e, 'response') and e.response:
                dprint(f"  Response: {e.response.text}")
            return False
    
    def get_albums(self):
        """Get list of existing albums - matches /albums GET endpoint"""
        url = f"{self.api_base}/albums"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            dprint(f"✗ Failed to fetch albums: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"  Response: {e.response.text}")
            return []
    
    def create_album(self, album_name):
        """Create a new album - matches /albums POST endpoint"""
        url = f"{self.api_base}/albums"
        payload = {
            'albumName': album_name,
            'description': f'Imported from local folder on {datetime.now().strftime("%Y-%m-%d")}'
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            album = response.json()
            dprint(f"✓ Created album: {album_name} (ID: {album.get('id')})")
            return album
        except requests.exceptions.RequestException as e:
            dprint(f"✗ Failed to create album: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"  Response: {e.response.text}")
            return None
    
    def calculate_sha1(self, file_path):
        """Calculate SHA1 hash of file for duplicate detection"""
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha1.update(chunk)
        return sha1.hexdigest()
    
    def check_bulk_upload(self, files):
        """Check which files already exist using /assets/bulk-upload-check endpoint"""
        url = f"{self.api_base}/assets/bulk-upload-check"
        
        assets = []
        for file_path in files:
            checksum = self.calculate_sha1(file_path)
            assets.append({
                'id': str(file_path),
                'checksum': checksum
            })
        
        payload = {'assets': assets}
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            dprint(f"✗ Bulk upload check failed: {e}")
            return None

    def upload_photo_and_sidecar(self, file_path, sidecar_path, album_name=None):
        url = f"{self.api_base}/assets"
        filename = os.path.basename(file_path)
        dprint(f"upload_photo_and_sidecar() file_path:{file_path} sidecar_path:{sidecar_path} album_name:{album_name}")
        # Get or create album
        album_id = None
        if album_name:
            dprint(f"\nLooking for album: {album_name}")
            albums = self.get_albums()
            for album in albums:
                if album['albumName'].lower() == album_name.lower():
                    album_id = album['id']
                    dprint(f"✓ Found existing album: {album_name} (ID: {album_id})")
                    dprint(f"  Assets: {album.get('assetCount', 0)}")
                    break

            if not album_id:
                dprint(f"Album '{album_name}' not found, creating...")
                new_album = self.create_album(album_name)
                if new_album:
                    album_id = new_album['id']
                pass

        # Determine mime type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            # Default based on extension
            ext = Path(file_path).suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.heic': 'image/heic',
                '.heif': 'image/heif',
                '.avif': 'image/avif',
                '.tiff': 'image/tiff',
                '.bmp': 'image/bmp'
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')

        # Calculate SHA1 for duplicate detection
        checksum = self.calculate_sha1(file_path)

        # Create a unique device asset ID
        device_asset_id = f"{os.path.basename(file_path)}-{os.path.getsize(file_path)}-{checksum[:8]}"
        device_id = self.importer_name
        stat = os.stat(file_path)
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        created = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        dprint(f"mime_type:{mime_type}")
        # Prepare file for upload
        with open(file_path, "rb") as img, open(sidecar_path, "rb") as xmp:
            files = {
                'assetData': (os.path.basename(file_path), img, mime_type),
                'sidecarData' :(os.path.basename(sidecar_path), xmp, 'application/xml')
            }

            # Prepare metadata according to AssetMediaCreateDto
            data = {
                'deviceAssetId': device_asset_id,
                'deviceId': device_id,
                "fileCreatedAt": created,
                "fileModifiedAt": modified,
                'isFavorite': 'false'
            }

            # Add checksum header for server-side duplicate detection
            headers = {'x-immich-checksum': checksum}

            try:
                dprint(f"Uploading to {url}...")

                response = self.session.post(url, data=data, files=files, headers=headers)

                # Handle different response codes
                if response.status_code == 200:
                    result = response.json()
                    dprint(f"Duplicate (status: {result.get('status')})")
                    return {'duplicate': True, 'id': result.get('id')}
                elif response.status_code == 201:
                    result = response.json()
                    dprint(f"Uploaded (ID: {result.get('id')})")
                    uploaded_asset_ids = []
                    uploaded_asset_ids.append(result['id'])
                    asset_id = result.get("id")
                    status = result.get("status")

                    # Add assets to album if we have an album ID and uploaded assets
                    dprint(f"album_id:{album_id} uploaded_asset_ids:{uploaded_asset_ids}")
                    if album_id and uploaded_asset_ids:
                        dprint(f"\nAdding {len(uploaded_asset_ids)} assets to album '{album_name}'...")
                        self.add_assets_to_album(album_id, uploaded_asset_ids)
                    # update metadata
                    """
                    job_response = self.session.post(
                            f"{url}/jobs",
                            json={"assetIds": [asset_id], "name": "refresh-metadata"},
                            )
                    """
                    return {'success': True, 'id': result.get('id')}
                else:
                    dprint(f"Unexpected status code: {response.status_code}")
                    dprint(f"Response: {response.text}")
                    return None

            except requests.exceptions.RequestException as e:
                dprint(f"Failed: {e}")
                if hasattr(e, 'response') and e.response:
                    dprint(f"    Status: {e.response.status_code}")
                    dprint(f"    Response: {e.response.text}")
                return None

    def upload_photo(self, file_path, album_name=None, tags=None, rating=None, comfy_workflow=None):
        """Upload a single photo to Immich - matches /assets POST endpoint

        Args:
            file_path (str): Path to the image file
            album_name (str, optional): Name of album to add photo to
            tags (list, optional): List of tags to add to the photo
            rating (int, optional): Rating from 0-5 (0 means no rating)
            comfy_workflow (dict, optional): ComfyUI workflow data
        """
        filename = os.path.basename(file_path)
        dprint(f"filename:{filename}")
        # Validate rating if provided
        if rating is not None:
            if not isinstance(rating, int) or rating < 0 or rating > 5:
                raise ValueError(f"Rating must be an integer between 0 and 5. got {rating}")

        # Add rating to metadata if provided and > 0
        if rating and rating > 0:
            #data['rating'] = str(rating)  # Immich expects rating as string
            dprint(f"    Rating: {rating}/5")
        else:
            dprint(f"    Rating: Not set")


        # Create sidecar data if tags, rating, workflow are provided
        if tags or (rating and rating > 0) or comfy_workflow :
            if True:
                # Create new XMP with standard Dublin Core tags, rating, and our custom data
                xmp_template = f'''<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Immich Importer">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="{filename}"
        xmlns:dc="http://purl.org/dc/elements/1.1/"
        xmlns:xmp="http://ns.adobe.com/xap/1.0/"
        xmlns:gadzoinks="http://ns.gadzoinks.com/1.0/">
'''
                # System
                stat_info = os.stat(file_path)
                if False:
                    xmp_template += f'\n<System:FileName>{filename}</System:FileName>'
                    xmp_template += f'\n<System:Directory>.</System:Directory>'
                    xmp_template += f'\n<System:FileModifyDate>{datetime.utcfromtimestamp(stat_info.st_mtime).strftime("%Y:%m:%d %H:%M:%S+00:00")}</System:FileModifyDate>'
                    xmp_template += f'\n<System:FileAccessDate>{datetime.utcfromtimestamp(stat_info.st_atime).strftime("%Y:%m:%d %H:%M:%S+00:00")}</System:FileAccessDate>'
                #Add Dublin Core tags if provided (standard format for Immich)
                xmp_template += '\n<dc:format>image/png</dc:format>'
                if tags:
                    xmp_template += f'\n<dc:subject>\n <rdf:Bag>'
                    for tag in tags:
                        xmp_template += f'\n  <rdf:li>{tag}</rdf:li>'
                    xmp_template += f'\n </rdf:Bag>\n</dc:subject>'

                # Add standard XMP rating if provided
                if rating and rating > 0:
                    xmp_template += f'\n<xmp:Rating>{rating}</xmp:Rating>'
                # IPTC tags
                if False:
                    dt = datetime.fromtimestamp(stat_info.st_mtime)
                    xmp_template += f'\n<IPTC:DateCreated>{dt.strftime("%Y:%m:%d")}</IPTC:DateCreated>'
                    xmp_template += f'\n<IPTC:TimeCreated>{dt.strftime("%H:%M:%S")}</IPTC:TimeCreated>'
                    xmp_template += f'\n<IPTC:DigitalCreationDate>{dt.strftime("%Y:%m:%d")}</IPTC:DigitalCreationDate>'
                    xmp_template += f'\n<IPTC:DigitalCreationTime>{dt.strftime("%H:%M:%S")}</IPTC:DigitalCreationTime>'
                    if tags:
                        xmp_template += f'<IPTC:Keywords><rdf:Bag>'
                        for tag in tags:
                            xmp_template += f'<rdf:li>{tag}</rdf:li>'
                        xmp_template += f'</rdf:Bag></IPTC:Keywords>'

                # Gadzoinks
                if True:
                    # Add Gadzoinks tags (as CDATA) if provided
                    if tags:
                        xmp_template += f'\n<gadzoinks:tags><![CDATA[{json.dumps(tags)}]]></gadzoinks:tags>'
                    # Add Gadzoinks rating if provided
                    if rating and rating > 0:
                        xmp_template += f'\n<gadzoinks:rating>{rating}</gadzoinks:rating>\n'
                    # Add ComfyUI workflow if provided
                    if comfy_workflow:
                        xmp_template += f'''
        <gadzoinks:comfyui_workflow><![CDATA[{json.dumps(comfy_workflow)}]]></gadzoinks:comfyui_workflow>'''
                # Close XML
                xmp_template += '''
        </rdf:Description>
    </rdf:RDF>
  </x:xmpmeta>
<?xpacket end="w"?>
'''

                # Create temporary sidecar file
                temp_sidecar = Path(file_path).parent / f"{filename}.xmp"
                with open(temp_sidecar, 'w', encoding='utf-8') as f:
                    f.write(xmp_template)

                # Print sidecar contents for debugging
                dprint(f"\n    Creating sidecar file:")
                dprint(f"    {'='*50}")
                for line in xmp_template.split('\n'):
                    dprint(f"    {line}")
                dprint(f"    {'='*50}")
        try:
            self.upload_photo_and_sidecar(file_path, os.fspath(temp_sidecar)  , album_name=album_name)
        except Exception as e:
            print(f"exception:{e}")
        finally:
            # Clean up temporary sidecar file if we created one
            if temp_sidecar.exists():
                try:
                    temp_sidecar.unlink()
                except:
                    pass

    def add_assets_to_album(self, album_id, asset_ids):
        """Add assets to album - matches /albums/{id}/assets PUT endpoint"""
        if not asset_ids:
            return
        
        url = f"{self.api_base}/albums/{album_id}/assets"
        payload = {
            'ids': asset_ids
        }
        
        try:
            dprint(f"Adding {len(asset_ids)} assets to album...")
            response = self.session.put(url, json=payload)
            response.raise_for_status()
            results = response.json()
            
            successful = sum(1 for r in results if r.get('success'))
            failed = len(results) - successful
            
            if successful > 0:
                dprint(f"Added {successful} assets to album")
            if failed > 0:
                dprint(f"Failed to add {failed} assets to album")
                
        except requests.exceptions.RequestException as e:
            dprint(f"✗ Failed to add assets to album: {e}")
            if hasattr(e, 'response') and e.response:
                dprint(f"  Response: {e.response.text}")
    
    def import_photos(self, source_dir, album_name=None, recursive=False, extensions=None):
        """Import photos from a directory"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif', '.avif']
        
        source_path = Path(source_dir)
        if not source_path.exists():
            dprint(f"Source directory does not exist: {source_dir}")
            return
        
        # Test connection first
        if not self.test_connection():
            return
        
        # Get or create album
        album_id = None
        if album_name:
            dprint(f"\nLooking for album: {album_name}")
            albums = self.get_albums()
            for album in albums:
                if album['albumName'].lower() == album_name.lower():
                    album_id = album['id']
                    dprint(f"✓ Found existing album: {album_name} (ID: {album_id})")
                    dprint(f"  Assets: {album.get('assetCount', 0)}")
                    break
            
            if not album_id:
                dprint(f"Album '{album_name}' not found, creating...")
                new_album = self.create_album(album_name)
                if new_album:
                    album_id = new_album['id']
        
        # Find all photos
        dprint(f"\nScanning directory: {source_path}")
        if recursive:
            files = sorted([p for p in source_path.rglob('*') if p.is_file() and p.suffix.lower() in extensions])
            dprint(f"Recursive scan enabled")
        else:
            files = sorted([p for p in source_path.glob('*') if p.is_file() and p.suffix.lower() in extensions])
        
        if not files:
            dprint("✗ No photos found in the specified directory")
            return
        
        dprint(f"\nFound {len(files)} photos to import")
        dprint(f"First few files: {[f.name for f in files[:3]]}")
        
        # Upload photos
        successful = 0
        failed = 0
        skipped = 0
        uploaded_asset_ids = []
        
        for i, file_path in enumerate(files, 1):
            dprint(f"\nProcessing {i}/{len(files)}: {file_path.name}")
            result = self.upload_photo(file_path)
            
            if result:
                if result.get('duplicate'):
                    skipped += 1
                    if result.get('id'):
                        uploaded_asset_ids.append(result['id'])
                        dprint(f"    Added duplicate ID to album list")
                elif result.get('success'):
                    successful += 1
                    if result.get('id'):
                        uploaded_asset_ids.append(result['id'])
            else:
                failed += 1
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.2)
        
        dprint(f"\n{'='*60}")
        dprint(f"IMPORT COMPLETE")
        dprint(f"{'='*60}")
        dprint(f"  ✓ Successfully uploaded: {successful}")
        dprint(f"  ⏭️  Skipped (duplicates): {skipped}")
        dprint(f"  ✗ Failed: {failed}")
        dprint(f"  📊 Total processed: {len(files)}")
        
        # Add assets to album if we have an album ID and uploaded assets
        if album_id and uploaded_asset_ids:
            dprint(f"\nAdding {len(uploaded_asset_ids)} assets to album '{album_name}'...")
            self.add_assets_to_album(album_id, uploaded_asset_ids)
        
        if album_name:
            dprint(f"  📁 Album: {album_name}")

def list_albums(importer):
    """List all available albums using the /albums endpoint"""
    dprint(f"\nFetching albums from {importer.server_url}...")
    
    # Test connection first
    if not importer.test_connection():
        return
    
    albums = importer.get_albums()
    
    if not albums:
        dprint("No albums found or unable to fetch albums.")
        return
    
    print(f"\n{'='*60}")
    print(f"FOUND {len(albums)} ALBUM(S)")
    print(f"{'='*60}")
    
    for i, album in enumerate(albums, 1):
        album_id = album.get('id', 'N/A')
        album_name = album.get('albumName', 'Unnamed')
        asset_count = album.get('assetCount', 0)
        created_at = album.get('createdAt', 'Unknown')
        description = album.get('description', '')
        is_shared = album.get('shared', False)
        
        # Format the date if it exists
        if created_at != 'Unknown' and created_at:
            try:
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            except:
                pass
        
        print(f"\n{i}. 📁 {album_name}")
        print(f"   ID: {album_id}")
        print(f"   📸 Photos: {asset_count}")
        print(f"   📅 Created: {created_at}")
        if description:
            print(f"   📝 Description: {description}")
        if is_shared:
            print(f"   🔗 Shared: Yes")
        
        # Show owner if available
        owner = album.get('owner')
        if owner:
            print(f"   👤 Owner: {owner.get('name', 'Unknown')}")
    
    print(f"\n{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description='Import local photos to Immich server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all albums
  %(prog)s --list-albums --api-key YOUR_API_KEY
  
  # Import photos from directory
  %(prog)s /path/to/photos --api-key YOUR_API_KEY
  
  # Import recursively and add to album
  %(prog)s /path/to/photos --api-key YOUR_API_KEY --album "Vacation 2024" --recursive
  
  # Login with email/password
  %(prog)s /path/to/photos --email user@example.com --password yourpassword
        """
    )
    
    # Make source optional when listing albums
    parser.add_argument('source', nargs='?', help='Source directory containing photos')
    parser.add_argument('--server', default='http://redstar.local:2283', 
                       help='Immich server URL (default: http://redstar.local:2283)')
    parser.add_argument('--api-key', help='Immich API key (get from user settings)')
    parser.add_argument('--email', help='Immich login email (if no API key)')
    parser.add_argument('--password', help='Immich login password (if no API key)')
    parser.add_argument('--album', help='Album name to add photos to')
    parser.add_argument('--list-albums', action='store_true', 
                       help='List all available albums')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='Search for photos recursively in subdirectories')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif', '.avif'],
                       help='File extensions to import (default: common image formats)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.list_albums and not args.source:
        parser.error("Source directory is required when not listing albums")
    
    # Create importer instance
    importer = ImmichImporter(args.server, args.api_key)
    
    # Login if API key not provided
    if not args.api_key:
        if not args.email or not args.password:
            print("Error: Either provide --api-key or both --email and --password")
            sys.exit(1)
        
        print(f"\nLogging in to {args.server}...")
        if not importer.login(args.email, args.password):
            sys.exit(1)
    else:
        print(f"\nUsing API key for {args.server}")
    
    # Check if we're listing albums
    if args.list_albums:
        list_albums(importer)
        return
    
    # Import photos
    importer.import_photos(args.source, args.album, args.recursive, args.extensions)

#if __name__ == "__main__":
#    main()

