import React, { useState, lazy, Suspense } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  Box,
  CircularProgress,
  Paper,
  Divider,
  TextField,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Skeleton,
  Fade,
  Zoom,
  useTheme,
  alpha
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ReportProblemIcon from '@mui/icons-material/ReportProblem';
import axios from 'axios';

// Lazy load the Report component
const Report = lazy(() => import('./Report'));

function Navbar() {
  return (
    <AppBar position="static" elevation={0} sx={{ background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)' }}>
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
          AI Resume Analyzer
        </Typography>
        <Button color="inherit" sx={{ '&:hover': { backgroundColor: alpha('#fff', 0.1) } }}>Home</Button>
        <Button color="inherit" sx={{ '&:hover': { backgroundColor: alpha('#fff', 0.1) } }}>About</Button>
      </Toolbar>
    </AppBar>
  );
}

function LoadingSkeleton() {
  return (
    <Box sx={{ width: '100%', mt: 4 }}>
      <Skeleton variant="rectangular" height={200} sx={{ borderRadius: 2 }} />
      <Box sx={{ mt: 2 }}>
        <Skeleton variant="text" width="60%" height={40} />
        <Skeleton variant="text" width="40%" height={30} />
      </Box>
      <Box sx={{ mt: 2 }}>
        <Skeleton variant="rectangular" height={100} sx={{ borderRadius: 2 }} />
      </Box>
    </Box>
  );
}

export default function ResumeUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const theme = useTheme();

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!selectedFile || !jobDescription.trim()) return;
    setUploading(true);

    const formData = new FormData();
    formData.append('resume', selectedFile);
    formData.append('job_description', jobDescription);

    try {
      const response = await axios.post('http://localhost:8000/upload', formData);
      setResult(response.data);
      setUploading(false);
    } catch (error) {
      console.error('Upload failed:', error);
      setResult({ error: 'Upload failed. Please try again.' });
      setUploading(false);
    }
  };

  return (
    <div>
      <Navbar />
      <Container maxWidth="md">
        <Fade in={true} timeout={1000}>
          <Box textAlign="center" mt={5}>
            <Zoom in={true} timeout={1000}>
              <Typography 
                variant="h4" 
                gutterBottom 
                sx={{ 
                  fontWeight: 'bold',
                  background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  mb: 4
                }}
              >
                Upload Resume and Job Description
              </Typography>
            </Zoom>

            <input
              type="file"
              accept=".pdf,.doc,.docx"
              onChange={handleFileChange}
              style={{ display: 'none' }}
              id="upload-input"
            />

            <label htmlFor="upload-input">
              <Button
                variant="contained"
                component="span"
                startIcon={<CloudUploadIcon />}
                sx={{ 
                  mt: 2,
                  background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                  boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #1976D2 30%, #1E88E5 90%)',
                  }
                }}
              >
                Choose Resume
              </Button>
            </label>

            {selectedFile && (
              <Fade in={true}>
                <Typography variant="body1" mt={2} sx={{ color: theme.palette.primary.main }}>
                  Selected: {selectedFile.name}
                </Typography>
              </Fade>
            )}

            <TextField
              label="Paste Job Description Here"
              placeholder="Looking for a DevOps engineer with AWS, Docker, and CI/CD..."
              multiline
              fullWidth
              rows={10}
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              sx={{ 
                mt: 3,
                '& .MuiOutlinedInput-root': {
                  '&:hover fieldset': {
                    borderColor: theme.palette.primary.main,
                  },
                },
              }}
            />

            <Button
              variant="contained"
              color="primary"
              onClick={handleUpload}
              sx={{ 
                mt: 3,
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                boxShadow: '0 3px 5px 2px rgba(33, 203, 243, .3)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #1976D2 30%, #1E88E5 90%)',
                }
              }}
              disabled={!selectedFile || !jobDescription || uploading}
            >
              {uploading ? <CircularProgress size={24} color="inherit" /> : 'Analyze Match'}
            </Button>

            {result && result.error && (
              <Fade in={true}>
                <Typography color="error" mt={4}>{result.error}</Typography>
              </Fade>
            )}

            {result && result.score && (
              <Suspense fallback={<LoadingSkeleton />}>
                <Fade in={true} timeout={1000}>
                  <Report result={result} />
                </Fade>
              </Suspense>
            )}
          </Box>
        </Fade>
      </Container>
    </div>
  );
}
