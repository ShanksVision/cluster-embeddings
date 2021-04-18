using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using ViDi2.Training.UI;
using ViDi2.Training;
using ViDi2;

public class Script
{
    static string basePath = @"C:\Projects\store";    
    
    public static void Main(ViDi2.Training.ITool tool)
    {
        bool scriptResult = false;
        if (tool == null)
        {
            Console.WriteLine("No Tool selected");
            return;
        }
        
        //Create a directory to store view images
        var dir = Directory.CreateDirectory(string.Format("{0}\\{1}", basePath, tool.Name));
        basePath = dir.FullName;        
       
        //Save views        
        IGreenDatabase dbase = tool.Database as IGreenDatabase;
        //SaveViews(dbase, false);  
        
        //Run Auto label suggestions
        scriptResult = RunScript(basePath, 30);       
        
        //Import lableing
        if(scriptResult)
            ImportLabels("pred.csv", dbase);
        
    }
    
    public static void SaveViews(IGreenDatabase db, bool isLabeled, string filter="all")
    {        
        string className = "", path = basePath;
        int dbIndex = 0; 
        ICollection<SortedViewKey> dbViews = db.List("all");  
        int totalCount = dbViews.Count, showLogEvery = 20;
        //EncoderParameters enc = new EncoderParameters(1);       
        //var depth = new EncoderParameter(System.Drawing.Imaging.Encoder.ColorDepth, 24L);
        //enc.Param[0] = depth;
        //var codec = ImageCodecInfo.GetImageEncoders().Where(x => x.MimeType == "image/tiff").First();

        if(isLabeled)
        {
            foreach(var tag in ((ViDi2.IGreenTool)db.Tool).KnownTags)
               Directory.CreateDirectory(string.Format("{0}\\{1}", basePath, tag)); 
        }
        else
        {
            var dir = Directory.CreateDirectory(string.Format("{0}\\{1}", basePath, "data"));
            path = dir.FullName;
        }
            
      
        foreach(SortedViewKey k in dbViews)
        {        
            string img_path;        
            IImage img = db.GetViewImage(k);
            if(isLabeled)
            {
                className = db.GetMarking(k.SampleName).Views[k.Index].Tags[0].Name;
                img_path = string.Format("{0}\\{1}\\{2}_{3}.png",path,className,Path.GetFileNameWithoutExtension(k.SampleName),k.Index);                
            }
            else
                img_path = string.Format("{0}\\{1}_{2}.png",path,Path.GetFileNameWithoutExtension(k.SampleName),k.Index);
           
                                   
            img.Bitmap.Save(img_path, System.Drawing.Imaging.ImageFormat.Png);
            dbIndex++;             
            if(dbIndex % showLogEvery == 0 || dbIndex == totalCount)
                Console.WriteLine("Images Saved = {0}/{1}..", dbIndex, totalCount);       
                      
        }
    }
    
    public static bool RunScript(string path, int classes=2)
    {
        string workingDir = @"C:\DL_CV\MyProjects";
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    RedirectStandardInput = true,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = workingDir                
                }
            };
            process.Start();
            using (var swriter = process.StandardInput)
            {
                if (swriter.BaseStream.CanWrite)
                {
                    //activate Anaconda
                    swriter.WriteLine(@"C:\Users\cupertino_user\anaconda3\Scripts\activate.bat");
                    // Activate environment
                    swriter.WriteLine("activate allin1");               
                    // run py script
                    swriter.WriteLine(string.Format("python Cluster_embeddings.py --path {0} --classes {1}", path, classes));
                }
            }
            
            // read multiple output lines
            bool success = false;
            using(var sreader = process.StandardOutput)
            {
                while (!sreader.EndOfStream)
                {
                    var line = sreader.ReadLine();
                    Console.WriteLine(line);
                    if(line.Contains("Program exited"))
                        success = line.Contains("1");
                        
                }
            }

            return success;
        }
        catch(System.Exception e)
        {
            Console.WriteLine("Error in running script");
            Console.WriteLine(e.Message);
            return false;
        }
        
    }
    
    public static void ImportLabels(string filename, IGreenDatabase gdbase)
    {
        Console.WriteLine("Importing labels into Cognex DL....");
        StreamReader sr = null;
        string line, extension=".png";
        try
        {        
            sr = new StreamReader(string.Format("{0}\\{1}", basePath, filename));
            while((line = sr.ReadLine()) != null)
            {
                var imgAndLabels = line.Split(',');
                var filter = Path.GetFileNameWithoutExtension(imgAndLabels[0]);
                filter = string.Format("filename=\'{0}{1}\' and view_index={2}", filter.Substring(0, filter.Length-2), extension, filter[filter.Length-1]);
                //Console.WriteLine(filter);                 
                gdbase.Tag(filter, imgAndLabels[1]);    
            }
            
            gdbase.Update("all", false, true);         
            Console.WriteLine("Labels Imported Successfully");
        }
        catch(System.Exception e)
        {
            Console.WriteLine(e.Message);
        }
        finally
        {
            if(sr != null)
            {
                sr.Close();
                sr.Dispose();                
            }
        }  
    }    
    
}