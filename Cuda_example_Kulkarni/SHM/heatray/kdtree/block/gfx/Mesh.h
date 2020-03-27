/*
   Filename : Mesh.h
   Author   : Cody White
   Version  : 2.0

   Purpose  : Class to define a triangle mesh. 

   Change List:

      - 12/21/2009  - Created (Cody White)
	  - 01/27/2010  - Templated the class (Cody White)
	  - 02/26/2010  - Added texturing to the mesh.
	  - 02/07/2011  - Added MeshParser support for different file types.
	  - 02/09/2011  - Converted the parser to use Assimp for model file parsing.
*/

#pragma once

#include <gfx/VertexBuffer.h>
#include <gfx/Material.h>
#include <gfx/Triangle.h>
#include <assimp.hpp>
#include <aiScene.h>
#include <aiPostProcess.h>
#include <util/string.h>
#include <vector>
#include <map>
#include <iostream>

namespace gfx
{

class Mesh
{
	public:

		/**
		  * Default constructor.
		  */
		Mesh (void)
		{
			m_flags =  aiProcess_Triangulate;
			m_flags |= aiProcess_GenSmoothNormals;
				      //aiProcess_JoinIdenticalVertices |
					  //aiProcess_SortByPType;//			  |
					  //aiProcess_GenSmoothNormals	  |
					  //aiProcess_FixInfacingNormals;
			m_use_tangents = false;
		}

		/**
		  * Destructor.
		  */
		~Mesh (void)
		{
			clearCPUData ();
			m_tangent_loc.clear ();
			m_bitangent_loc.clear ();
		}

		/**
		  * Calculate tangents.
		  * Must be called before load () to have any effect.
		  */
		void useTangents (void)
		{
			m_use_tangents = true;
			m_flags |= aiProcess_CalcTangentSpace;
		}

		/**
		  * Set the tangent and bitangent locations in a program.
		  */
		void setNormalMapParams (GLint tangent_loc, GLint bitangent_loc, int context_id)
		{
			m_tangent_loc[context_id] = tangent_loc;
			m_bitangent_loc[context_id] = bitangent_loc;
		}

		/**
		  * Load a mesh and store it into a list of triangles.
		  * @param filename Path to the file.
		  * @param scale Scale value for the mesh.
		  * @param create_vbo Automatically create the vbos for the objects.
		  */
		bool load (util::string filename, bool create_vbo = false, float scale = 1.0f)
		{
			Assimp::Importer importer;
			const aiScene *scene = importer.ReadFile (filename, m_flags);
			if (!scene)
			{
				std::cout << "Mesh::load () - Enable to load " << filename << std::endl;
				std::cout << "Assimp Error: " << importer.GetErrorString () << std::endl;
				return false;
			}
			
			// Figure out the path without the filename.
			util::string path = filename;
			path = path.substr (0, path.find_last_of ("/") + 1);

			// Get the triangles.
			m_meshes.resize (scene->mNumMeshes);
			for (size_t i = 0; i < scene->mNumMeshes; ++i)
			{
				aiMesh *mesh = scene->mMeshes[i];
				loadMaterial (scene, mesh->mMaterialIndex, m_meshes[i].material, path);

				for (size_t f = 0; f < mesh->mNumFaces; ++f)
				{
					gfx::Triangle tri;
					aiFace *face = &mesh->mFaces[f];
					for (size_t j = 0; j < face->mNumIndices; ++j)
					{
						int index = face->mIndices[j];
						tri.m_vertices[j] = mesh->mVertices[index] * scale;
						if (mesh->HasNormals ())
						{
							tri.m_normals[j] = mesh->mNormals[index];
						}
						if (mesh->HasTextureCoords (0))
						{
							tri.m_texture_coords[j] = mesh->mTextureCoords[0][index];
						}
						if (mesh->HasTangentsAndBitangents ())
						{
							tri.m_tangents[j] = mesh->mTangents[index];
							tri.m_bitangents[j] = mesh->mBitangents[index];
						}
					}
					
					tri.m_material = &(m_meshes[i].material);
					m_meshes[i].triangles.push_back (tri);
				}
			}

			if (create_vbo)
			{
				initContext (0);
			}

			return true;
		}

		/**
		  * Initialize the context for use.
		  * @param context_id ID for the context.
		  */
		void initContext (int context_id)
		{
			if (m_use_tangents == false)
			{
				m_sizeof_render_data = sizeof (RenderData);
			}
			else
			{
				m_sizeof_render_data = sizeof (NormalRenderData);
			}

			// Allocate a VBO for each material found.
			m_vbos.resize (m_meshes.size ());
			int vbo_count = 0;
			for (size_t i = 0; i < m_meshes.size (); ++i)
			{
				if (m_use_tangents == false)
				{
					populateVBO (m_meshes[i], vbo_count++, context_id);
				}
				else
				{
					populateNormalMapVBO (m_meshes[i], vbo_count++, context_id);
				}
			}

			initializeTextures (context_id);
		}

		/**
		  * Destroy the context.
		  * @param context_id ID for the context.
		  */
		void destroyContext (int context_id)
		{
			// Remove the VBO's
			for (size_t i = 0; i < m_vbos.size (); ++i)
			{
				m_vbos[i].vbo.destroyContext (context_id);
			}

			// Remove the textures
			for (size_t i = 0; i < m_meshes.size (); ++i)
			{
				m_meshes[i].material.texture.destroyContext (context_id);
			}

			//m_tangent_loc.remove (context_id);
			//m_bitangent_loc.remove (context_id);
		}

		/**
		  * Initialize the textures found for use.
		  * @param context_id ID for this context.
		  */
		void initializeTextures (int context_id)
		{
			for (size_t i = 0; i < m_meshes.size (); ++i)
			{
				if (m_meshes[i].material.texture.hasData ())
				{
					m_meshes[i].material.texture.initContext (context_id);
				}
			}
		}

		/**
		  * Clear the triangle lists.  If a vbo has been created, the internal
		  * lists can be cleared if no triangle data is needed.
		  */
		void clearCPUData (void)
		{
			for (size_t i = 0; i < m_meshes.size (); ++i)
			{
				m_meshes[i].material.texture.clearCPUData ();
				m_meshes[i].triangles.clear ();
			}
		}

		/**
		  * Render the mesh.
		  * @param context_id Id of the current context to render to.
		  */
		void render (int context_id) const 
		{
			if (m_vbos.size ())
			{
				// A VBO has been created for this mesh, so use it.
				gfx::VertexBuffer::enableVertexArrays (true);
				gfx::VertexBuffer::enableNormalArrays (true);
				gfx::VertexBuffer::enableTexCoordArrays (true);

				if (m_use_tangents)
				{
					gfx::VertexBuffer::enableVertexAttrib (m_tangent_loc[context_id], true);
					gfx::VertexBuffer::enableVertexAttrib (m_bitangent_loc[context_id], true);
				}

				for (size_t i = 0; i < m_vbos.size (); ++i)
				{
					bool bind_texture = false;
					applyMaterial (m_vbos[i].material);

					// Apply any texture that this material might have.
					bind_texture = m_vbos[i].material->texture.valid (context_id);

					if (bind_texture)
					{
						m_vbos[i].material->texture.bind (context_id);
					}

					m_vbos[i].vbo.bind (context_id);
					m_vbos[i].vbo.setVertexPointer (3, GL_FLOAT, m_sizeof_render_data, 0);
					m_vbos[i].vbo.setNormalPointer (GL_FLOAT, m_sizeof_render_data, sizeof (math::vec3f));
					m_vbos[i].vbo.setTexCoordPointer (2, GL_FLOAT, m_sizeof_render_data, 2 * sizeof (math::vec3f));

					if (m_use_tangents)
					{
						m_vbos[i].vbo.setAttribPointer (m_tangent_loc[context_id], 3, GL_FLOAT, false, m_sizeof_render_data, (2 * sizeof (math::vec3f)) + sizeof (math::vec2f));
						m_vbos[i].vbo.setAttribPointer (m_bitangent_loc[context_id], 3, GL_FLOAT, false, m_sizeof_render_data, (3 * sizeof (math::vec3f)) + sizeof (math::vec2f));
					}

					m_vbos[i].vbo.render (GL_TRIANGLES, 0, m_vbos[i].num_vertices);
					m_vbos[i].vbo.unbind ();

					if (bind_texture)
					{
						m_vbos[i].material->texture.unbind ();
					}
				}

				gfx::VertexBuffer::enableVertexArrays (false);
				gfx::VertexBuffer::enableNormalArrays (false);
				gfx::VertexBuffer::enableTexCoordArrays (false);

				if (m_use_tangents)
				{
					gfx::VertexBuffer::enableVertexAttrib (m_tangent_loc[context_id], false);
					gfx::VertexBuffer::enableVertexAttrib (m_bitangent_loc[context_id], false);
				}

				return;	
			}

			// Else render in immediate mode.
			for (size_t i = 0; i < m_meshes.size (); ++i)
			{
				bool bind_texture = false;
				applyMaterial (&m_meshes[i].material);

				// Apply any texture that this material might have.
				bind_texture = m_meshes[i].material.texture.valid (context_id);

				if (bind_texture)
				{
					m_meshes[i].material.texture.bind (context_id);
				}

				glBegin (GL_TRIANGLES);
				for (size_t j = 0; j < m_meshes[i].triangles.size (); ++j)
				{
					for (int k = 0; k < 3; ++k)
					{
						//if (m_use_normals)
						{
							glNormal3f (m_meshes[i].triangles[j].m_normals[k].x (),
										m_meshes[i].triangles[j].m_normals[k].y (),
										m_meshes[i].triangles[j].m_normals[k].z ());
						}

						//if (m_use_texture)
						{
							glTexCoord2f (m_meshes[i].triangles[j].m_texture_coords[k].x (),
										  m_meshes[i].triangles[j].m_texture_coords[k].y ());
						}

						glVertex3f (m_meshes[i].triangles[j].m_vertices[k].x (),
									m_meshes[i].triangles[j].m_vertices[k].y (),
									m_meshes[i].triangles[j].m_vertices[k].z ());
					}
				}
				glEnd ();

				if (bind_texture)
				{
					m_meshes[i].material.texture.unbind ();
				}
			}
		}

		/**
		  * Render debug information for this model.
		  */
		void renderDebug (void) const
		{
			glDisable (GL_LIGHTING);
			for (size_t i = 0; i < m_meshes.size (); ++i)
			{
				glBegin (GL_LINES);
				for (size_t j = 0; i < m_meshes[i].triangles.size (); ++j)
				{
					for (int k = 0; k < 3; ++k)
					{
						math::vec3f vertex = m_meshes[i].triangles[j].m_vertices[k];
						math::vec3f normal = m_meshes[i].triangles[j].m_normals[k];
						math::vec3f tangent = m_meshes[i].triangles[j].m_tangents[k];
						math::vec3f bitangent = m_meshes[i].triangles[j].m_bitangents[k];

						// Draw the normal.
						glColor3f (1.0f, 0.0f, 0.0f);
						glVertex3fv (vertex.v);
						glVertex3fv ((vertex + normal * 0.2f).v);

						// Draw the tangent.
						glColor3f (0.0f, 1.0f, 0.0f);
						glVertex3fv (vertex.v);
						glVertex3fv ((vertex + tangent * 0.2f).v);

						// Draw the bitangent.
						glColor3f (0.0f, 0.0f, 1.0f);
						glVertex3fv (vertex.v);
						glVertex3fv ((vertex + bitangent * 0.2f).v);
					}
				}
				glEnd ();
			}
			glEnable (GL_LIGHTING);
		}

		/**
		  * Submesh piece.
		  */
		struct MeshPiece
		{
			/**
			  * Default constructor.
			  */
			MeshPiece (void)
			{
			}

			/**
			  * Copy constructor.
			  */
			MeshPiece (const MeshPiece &other)
			{
				material = other.material;
				triangles = other.triangles;
			}

			gfx::Material material;					// Material associated with this piece.
			std::vector <gfx::Triangle > triangles;	// Triangles associated with this piece.
		};

		/**
		  * Get the number of meshes contained within this model file.
		  */
		const size_t getNumMeshes (void) const
		{
			return m_meshes.size ();
		}

		const size_t getNumTriangles (void) const
		{
			size_t cnt = 0;
			for (int i = 0; i < m_meshes.size(); i++) {
				cnt += m_meshes[i].triangles.size();
			}
			return cnt;
		}

		/**
		  * Get access to a particular mesh contained within this class.
		  */
		MeshPiece * getMesh (const size_t i)
		{
			return &m_meshes[i];
		}

	private:

		// Data for rendering from a VBO.
		struct RenderData
		{
			math::vec3f vertex;
			math::vec3f normal;
			math::vec2f tex_coord;
		};

		// Data for rendering from a VBO with normal map data.
		struct NormalRenderData
		{
			math::vec3f vertex;
			math::vec3f normal;
			math::vec2f tex_coord;
			math::vec3f tangent;
			math::vec3f bitangent;
		};

		// Data for a VBO.
		struct VBOData
		{
			gfx::VertexBuffer vbo;		// VBO to use for rendering.
			size_t num_vertices;		// Number of vertices in the VBO.
			Material *material;	// Reference to the material to apply for the triangles in this vbo.
		};

		/**
		  * Populate a VBO with triangle data.
		  * @param iter Iterator to the proper element in the m_triangles map.
		  * @param vbo_count VBO index to populate.
		  * @param context_id ID for this context.
		  */
		void populateVBO (MeshPiece &piece, int vbo_count, int context_id)
		{
			std::vector <RenderData> data;
			data.resize (piece.triangles.size () * 3); // Times 3 because the size is the number of triangles, each made up of 3 pieces of data.
			int data_counter = 0;
			// Loop over all of the triangles.
			for (size_t i = 0; i < piece.triangles.size (); ++i)
			{
				// Loop over each piece of data in the triangle.
				for (size_t j = 0; j < 3; ++j)
				{
					data[data_counter].vertex = piece.triangles[i].m_vertices[j];
					data[data_counter].normal = piece.triangles[i].m_normals[j];
					data[data_counter].tex_coord = piece.triangles[i].m_texture_coords[j];
					data_counter++;
				}
			}

			m_vbos[vbo_count].num_vertices = data.size ();
			m_vbos[vbo_count].material = &piece.material;
			m_vbos[vbo_count].vbo.load ((void *)&data[0], m_sizeof_render_data * data.size (), context_id);
		}

		/**
		  * Populate a VBO with triangle data that contains information for tangent space.
		  * @param iter Iterator to the proper element in the m_triangles map.
		  * @param vbo_count VBO index to populate.
		  * @param context_id ID for this context.
		  */
		void populateNormalMapVBO (MeshPiece &piece, int vbo_count, int context_id)
		{
			std::vector <NormalRenderData> data;
			data.resize (piece.triangles.size () * 3); // Times 3 because the size is the number of triangles, each made up of 3 pieces of data.
			int data_counter = 0;
			// Loop over all of the triangles.
			for (size_t i = 0; i < piece.triangles.size (); ++i)
			{
				// Loop over each piece of data in the triangle.
				for (size_t j = 0; j < 3; ++j)
				{
					data[data_counter].vertex = piece.triangles[i].m_vertices[j];
					data[data_counter].normal = piece.triangles[i].m_normals[j];
					data[data_counter].tex_coord = piece.triangles[i].m_texture_coords[j];
					data[data_counter].tangent = piece.triangles[i].m_tangents[j];
					data[data_counter].bitangent = piece.triangles[i].m_bitangents[j];
					data_counter++;
				}
			}

			m_vbos[vbo_count].num_vertices = data.size ();
			m_vbos[vbo_count].material = &piece.material;
			m_vbos[vbo_count].vbo.load ((void *)&data[0], m_sizeof_render_data * data.size (), context_id);
		}

		/**
		  * Load a specific material from the assimp mesh.
		  */
		void loadMaterial (const aiScene *scene, size_t material_index, gfx::Material &material, util::string base_path)
		{
			aiMaterial *mat = scene->mMaterials[material_index];
			aiColor3D color;

			// Get the diffuse component of the material.
			mat->Get (AI_MATKEY_COLOR_DIFFUSE, color);
			material.diffuse = math::vec3f (color.r, color.g, color.b);
			
			// Get the specular component of the material.
			mat->Get (AI_MATKEY_COLOR_SPECULAR, color);
			material.specular = math::vec3f (color.r, color.g, color.b);

			// Get the ambient component of the material.
			mat->Get (AI_MATKEY_COLOR_AMBIENT, color);
			material.ambient = math::vec3f (color.r, color.g, color.b);

			// Get the specular coefficient of the material.
			mat->Get (AI_MATKEY_SHININESS, material.specular_exponent);

			// Get the index of refraction of the material.
			mat->Get (AI_MATKEY_REFRACTI, material.index_of_refraction);

			// Get the transparent component of the material.
			mat->Get (AI_MATKEY_COLOR_TRANSPARENT, color);
			material.transmissive = math::vec3f (color.r, color.g, color.b);

			// Check to see if there's a diffuse texture.
			if (mat->GetTextureCount (aiTextureType_DIFFUSE) > 0)
			{
				aiString texture_path;
				mat->GetTexture (aiTextureType_DIFFUSE, 0, &texture_path);
				material.texture.texture_name = base_path + texture_path.data;

				// Load the texture.
				material.texture.load (material.texture.texture_name.c_str ());
			}

			// Get the name of the material.
			aiString string;
			mat->Get (AI_MATKEY_NAME, string);
			material.name = string.data;
		}

		/**
		  * Apply a material.
		  * @param material Material to apply.
		  */
		void applyMaterial (const Material *material) const
		{
			setMaterial(GL_AMBIENT, material->ambient);
			setMaterial(GL_DIFFUSE, material->diffuse);
			setMaterial(GL_SPECULAR, material->specular);
			setMaterial(GL_SHININESS, material->specular_exponent);
		}

		/**
		  * Set the current OpenGL state to use this material.
		  * @param param OpenGL parameter to set.
		  * @param v Value to set for the current parameter.
		  */
		inline void setMaterial (GLenum param, const math::vec3f &v) const
		{
			GLfloat q[4];
			q[0] = v.x();
			q[1] = v.y();
			q[2] = v.z();
			q[3] = 1.0;
			glMaterialfv(GL_FRONT_AND_BACK, param, q);
		}

		// Member variables.
		std::vector <VBOData>													m_vbos;					// List of VBOs to use for rendering
		unsigned int															m_flags;				// Bitfield to tell Assimp how we want to load the mesh
		bool																	m_use_tangents;			// Flag to know if we're using tangents or not
		std::vector <MeshPiece>													m_meshes;				// Vector of pieces of this mesh.
		int 																	m_sizeof_render_data;	// Sizeof the render data struct in bytes
		gfx::ContextBuffer<GLint>												m_tangent_loc;			// Location of the tangent variable in the shader.
		gfx::ContextBuffer<GLint>												m_bitangent_loc;		// Location of the bitangent variable in the shader.

};

} // end namespace

